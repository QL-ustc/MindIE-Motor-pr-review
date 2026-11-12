// Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
// MindIE is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//         http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

//! Worker registry for the KV conductor.
//!
//! Manages instance registration, unregistration, ZMQ subscriptions for
//! Mooncake-type publishers, and provides the query interface.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock as ParkingRwLock;

use crate::backend::{MatchMode, StoreBackend};
use crate::error::KvConductorError;
use crate::indexer::{CacheMaintenanceConfig, Indexer, IndexerKey};
use crate::protocols::*;

/// Extract the IP/host portion from a ZMQ endpoint URL.
/// e.g. "tcp://10.0.0.1:5557" → "10.0.0.1"
///      "tcp://[::1]:5557"   → "::1"
fn extract_ip_from_endpoint(endpoint: &str) -> Option<String> {
    let without_prefix = endpoint.strip_prefix("tcp://")?;
    // IPv6 bracket notation: tcp://[::1]:5557
    if let Some(rest) = without_prefix.strip_prefix('[') {
        let close_bracket = rest.find(']')?;
        return Some(rest[..close_bracket].to_string());
    }
    // IPv4 or hostname: tcp://10.0.0.1:5557
    let colon_pos = without_prefix.rfind(':')?;
    Some(without_prefix[..colon_pos].to_string())
}

/// Add `(instance_id, dp_rank)` entries to `hbm_ip_index` for each NPU endpoint.
fn add_hbm_ip_index_entries(
    ip_index: &HbmIpIndex,
    medium_endpoints: &HashMap<String, String>,
    instance_id: &str,
    dp_rank: u32,
) {
    for (medium_str, ep_url) in medium_endpoints {
        if !StorageMedium::is_hbm_key(medium_str) {
            continue;
        }
        let Some(ref ip) = extract_ip_from_endpoint(ep_url) else {
            continue;
        };
        let target = (instance_id.to_string(), dp_rank);
        let mut index = ip_index.write();
        let dps = index.entry(ip.clone()).or_default();
        let mut found = false;
        dps.retain(|candidate| {
            if candidate != &target {
                return true;
            }
            if found {
                return false;
            }
            found = true;
            true
        });
        if !found {
            dps.push(target);
        }
        tracing::info!(instance_id = %instance_id, dp_rank, ip = %ip, "HBM IP indexed for pool auto-attach");
    }
}

fn hbm_ips(medium_endpoints: &HashMap<String, String>) -> HashSet<String> {
    medium_endpoints
        .iter()
        .filter(|(medium, _)| StorageMedium::is_hbm_key(medium))
        .filter_map(|(_, endpoint)| extract_ip_from_endpoint(endpoint))
        .collect()
}

/// For each NPU endpoint in `medium_endpoints`, remove the (instance_id, dp_rank)
/// entry from `hbm_ip_index`. If a given IP has no remaining DPs, the IP key is
/// dropped entirely.
fn remove_hbm_ip_index_entries(
    ip_index: &HbmIpIndex,
    medium_endpoints: &HashMap<String, String>,
    instance_id: &str,
    dp_rank: u32,
) {
    for (medium_str, ep_url) in medium_endpoints {
        if !StorageMedium::is_hbm_key(medium_str) {
            continue;
        }
        let Some(ref ip) = extract_ip_from_endpoint(ep_url) else {
            continue;
        };
        let mut idx = ip_index.write();
        if let Some(dps) = idx.get_mut(ip.as_str()) {
            dps.retain(|(iid, dp)| iid != instance_id || dp != &dp_rank);
            if dps.is_empty() {
                idx.remove(ip.as_str());
            }
        }
        tracing::info!(instance_id = %instance_id, dp_rank, ip = %ip, "HBM IP removed from index");
    }
}

fn remove_stale_hbm_ip_index_entries(
    ip_index: &HbmIpIndex,
    old_medium_endpoints: &HashMap<String, String>,
    retained_ips: &HashSet<String>,
    instance_id: &str,
    dp_rank: u32,
) {
    let stale_endpoints = old_medium_endpoints
        .iter()
        .filter(|(medium, endpoint)| {
            StorageMedium::is_hbm_key(medium)
                && extract_ip_from_endpoint(endpoint).is_some_and(|ip| !retained_ips.contains(&ip))
        })
        .map(|(medium, endpoint)| (medium.clone(), endpoint.clone()))
        .collect();
    remove_hbm_ip_index_entries(ip_index, &stale_endpoints, instance_id, dp_rank);
}

/// Matching strategy and IP-index requirement for the ZMQ producers of a
/// registration.
///
/// Pool registrations (the Mooncake/Memcache cluster pool, and the YuanRong
/// node-level CPU/Disk pool) fan events out by node IP via `hbm_ip_index`;
/// engine/HBM registrations are tied to a single `(instance_id, dp_rank)`.
///
/// The live subscriber and `replay_events` must agree on both values for the
/// same registration — replay previously derived them from
/// `StoreBackend::match_mode()`, which keys on the backend rather than on the
/// registration shape and therefore returned `None` (no fan-out) for a
/// YuanRong node pool.
fn subscriber_match(is_pool: bool) -> (MatchMode, bool) {
    if is_pool {
        (MatchMode::IpOnly, true)
    } else {
        (MatchMode::None, false)
    }
}

/// Information about a registered endpoint for a worker.
#[derive(Debug, Clone, Serialize)]
pub struct EndpointInfo {
    /// Per-medium ZMQ PUB endpoints, e.g. {"npu": "tcp://...:5557", "cpu": "tcp://...:5558"}.
    pub medium_endpoints: HashMap<String, String>,
    pub engine_type: String,
    pub dp_rank: DpRank,
    pub replay_endpoint: Option<String>,
    /// Server-owned capability: NPU data is accepted only from an exact-DP
    /// subscriber path for this tuple. HTTP NPU ingestion permanently
    /// downgrades the capability until the tuple is re-registered.
    pub exact_dp_npu_provenance: bool,
}

/// A registered worker entry.
#[derive(Debug, Clone)]
pub struct WorkerEntry {
    pub instance_id: InstanceId,
    pub model_name: String,
    pub tenant_id: String,
    pub block_size: u32,
    pub store_backend: String,
    /// Per-DP backend identity. The public `store_backend` field retains its
    /// historical last-registration value for compatibility.
    store_backends: HashMap<DpRank, String>,
    /// dp_rank -> EndpointInfo
    pub endpoints: HashMap<DpRank, EndpointInfo>,
}

/// The central worker registry.
///
/// Thread-safe via internal locking (tokio::sync::RwLock for the inner
/// HashMap, protecting both registration metadata and indexer lifecycle).
pub struct WorkerRegistry {
    /// Instance ID -> WorkerEntry
    instances: tokio::sync::RwLock<HashMap<InstanceId, WorkerEntry>>,
    /// The shared indexer for all models/tenants
    indexer: Arc<Indexer>,
    /// Active ZMQ subscribers, keyed by (instance_id, dp_rank, endpoint_url).
    /// Multiple subscribers may exist per dp_rank when a backend uses separate
    /// ports for different storage media (e.g. YuanRong: HBM port vs DDR/SSD port).
    /// The endpoint URL in the key ensures each unique connection gets its own
    /// subscriber even when media share an endpoint (deduplication).
    zmq_subscribers: tokio::sync::RwLock<
        HashMap<(InstanceId, DpRank, String), crate::zmq_subscriber::ZmqSubscriber>,
    >,
    /// IP → list of (instance_id, dp_rank) lookup for Mooncake auto-attach.
    /// Built from HBM endpoint registrations; consumed by the pool subscriber
    /// to map `backend_id` in events to the correct DP(s).
    hbm_ip_index: HbmIpIndex,
    /// Serializes lifecycle transitions per instance while allowing unrelated
    /// instances to proceed during subscriber shutdown or replay.
    lifecycle_locks: DashMap<InstanceId, Arc<tokio::sync::Mutex<()>>>,
    /// Test seam for exercising registration metadata and IP-index lifecycle
    /// without creating background ZMQ tasks.
    subscriber_connections_enabled: bool,
}

impl WorkerRegistry {
    /// Resolve endpoint→media mapping from either the new `medium_endpoints`
    /// protocol or the legacy `endpoint` field (Mooncake Master compat).
    ///
    /// Returns a map of endpoint_url → Vec<StorageMedium>, deduplicated by URL.
    fn resolve_endpoint_media(
        req: &RegisterRequest,
    ) -> Result<HashMap<String, Vec<StorageMedium>>, KvConductorError> {
        // New protocol: per-medium endpoints
        if !req.medium_endpoints.is_empty() {
            let mut map: HashMap<String, Vec<StorageMedium>> = HashMap::new();
            for (medium_str, ep) in &req.medium_endpoints {
                let medium = StorageMedium::parse(medium_str);
                map.entry(ep.clone()).or_default().push(medium);
            }
            return Ok(map);
        }

        // Legacy protocol: single endpoint for all media (Mooncake Master)
        if let Some(ref ep) = req.endpoint {
            if ep.is_empty() {
                return Err(KvConductorError::Internal(
                    "endpoint is empty; either medium_endpoints or a valid endpoint is required"
                        .to_string(),
                ));
            }
            let mut map: HashMap<String, Vec<StorageMedium>> = HashMap::new();
            // Mooncake master publishes CPU and DISK events (never NPU) on one
            // port. We include NPU here so the indexer is ready for HBM events
            // from engine-level publishers that may share the same endpoint.
            map.insert(
                ep.clone(),
                vec![StorageMedium::Npu, StorageMedium::Cpu, StorageMedium::Disk],
            );
            return Ok(map);
        }

        // No endpoint configured — HTTP-events-only publisher.
        // The instance is registered for /events ingestion without ZMQ subscription.
        Ok(HashMap::new())
    }

    pub fn new() -> Self {
        Self::with_cache_config(CacheMaintenanceConfig::default())
    }

    pub fn with_cache_config(config: CacheMaintenanceConfig) -> Self {
        Self {
            instances: tokio::sync::RwLock::new(HashMap::new()),
            indexer: Arc::new(Indexer::with_config(config)),
            zmq_subscribers: tokio::sync::RwLock::new(HashMap::new()),
            hbm_ip_index: Arc::new(ParkingRwLock::new(HashMap::new())),
            lifecycle_locks: DashMap::new(),
            subscriber_connections_enabled: true,
        }
    }

    #[cfg(test)]
    fn without_subscriber_connections() -> Self {
        Self {
            instances: tokio::sync::RwLock::new(HashMap::new()),
            indexer: Arc::new(Indexer::new()),
            zmq_subscribers: tokio::sync::RwLock::new(HashMap::new()),
            hbm_ip_index: Arc::new(ParkingRwLock::new(HashMap::new())),
            lifecycle_locks: DashMap::new(),
            subscriber_connections_enabled: false,
        }
    }

    fn lifecycle_lock(&self, instance_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        Arc::clone(
            self.lifecycle_locks
                .entry(instance_id.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .value(),
        )
    }

    fn release_lifecycle_lock_if_unused(
        &self,
        instance_id: &str,
        lifecycle_lock: &Arc<tokio::sync::Mutex<()>>,
    ) {
        self.lifecycle_locks.remove_if(instance_id, |_, current| {
            Arc::ptr_eq(current, lifecycle_lock) && Arc::strong_count(current) == 2
        });
    }

    async fn take_subscribers(
        &self,
        instance_id: &str,
        dp_rank: u32,
    ) -> Vec<crate::zmq_subscriber::ZmqSubscriber> {
        let mut subscribers = self.zmq_subscribers.write().await;
        let matching: Vec<_> = subscribers
            .keys()
            .filter(|(iid, dp, _)| iid == instance_id && dp == &dp_rank)
            .cloned()
            .collect();
        matching
            .into_iter()
            .filter_map(|key| subscribers.remove(&key))
            .collect()
    }

    async fn shutdown_subscribers(subscribers: Vec<crate::zmq_subscriber::ZmqSubscriber>) {
        for subscriber in subscribers {
            subscriber.shutdown().await;
        }
    }

    /// Register a worker endpoint.
    ///
    /// Creates the indexer entry for the (model, tenant) pair if it doesn't
    /// exist yet. Spawns ZMQ subscribers for each unique per-medium endpoint
    /// declared in `medium_endpoints`.
    pub async fn register(&self, req: &RegisterRequest) -> Result<(), KvConductorError> {
        let endpoint_media = Self::resolve_endpoint_media(req)?;

        let sb = StoreBackend::parse(&req.store_backend);
        let is_pool =
            (sb.is_pool_auto_attach() && req.medium_endpoints.is_empty() && req.endpoint.is_some())
                || sb.is_node_pool_registration(&req.medium_endpoints);
        let exact_dp_npu_provenance = !is_pool
            && endpoint_media
                .values()
                .flatten()
                .any(|medium| *medium == StorageMedium::Npu);

        let lifecycle_lock = self.lifecycle_lock(&req.instance_id);
        let _lifecycle_guard = lifecycle_lock.lock().await;
        let old_registration = {
            let instances = self.instances.read().await;
            if let Some(existing) = instances.get(&req.instance_id) {
                if existing.model_name != req.modelname
                    || existing.tenant_id != req.tenant_id
                    || existing.block_size != req.block_size
                {
                    return Err(KvConductorError::RegistrationIdentityConflict {
                        instance_id: req.instance_id.clone(),
                        dp_rank: req.dp_rank,
                        reason: format!(
                            "registered model={} tenant={} block_size={}, requested model={} tenant={} block_size={}",
                            existing.model_name,
                            existing.tenant_id,
                            existing.block_size,
                            req.modelname,
                            req.tenant_id,
                            req.block_size
                        ),
                    });
                }
            }
            instances.get(&req.instance_id).and_then(|entry| {
                entry.endpoints.get(&req.dp_rank).map(|endpoint| {
                    let old_backend = entry
                        .store_backends
                        .get(&req.dp_rank)
                        .unwrap_or(&entry.store_backend)
                        .clone();
                    (old_backend, endpoint.medium_endpoints.clone())
                })
            })
        };
        let tuple_exists = old_registration.is_some();

        if let Some((old_backend, old_medium_endpoints)) = &old_registration {
            tracing::info!(
                instance_id = %req.instance_id,
                dp_rank = req.dp_rank,
                old_backend = %old_backend,
                new_backend = %req.store_backend,
                "re-registering existing worker"
            );

            let old_subscribers = self.take_subscribers(&req.instance_id, req.dp_rank).await;
            Self::shutdown_subscribers(old_subscribers).await;

            let retained_hbm_ips = if sb.index_hbm_ip() && !is_pool {
                hbm_ips(&req.medium_endpoints)
            } else {
                HashSet::new()
            };
            remove_stale_hbm_ip_index_entries(
                &self.hbm_ip_index,
                old_medium_endpoints,
                &retained_hbm_ips,
                &req.instance_id,
                req.dp_rank,
            );

            if let Some(indexer_entry) = self.indexer.get(&req.modelname, &req.tenant_id) {
                if old_backend.eq_ignore_ascii_case(&req.store_backend) {
                    indexer_entry.remove_worker_hbm(&req.instance_id, req.dp_rank);
                } else {
                    indexer_entry.remove_worker_all_media(&req.instance_id, req.dp_rank);
                }
            }
        }

        if sb.index_hbm_ip() && !is_pool {
            add_hbm_ip_index_entries(
                &self.hbm_ip_index,
                &req.medium_endpoints,
                &req.instance_id,
                req.dp_rank,
            );
        }

        // Resolve routing once so live ingestion and replay use the same identity.
        // YuanRong node-pool subscribers must look up hbm_ip_index by node
        // IP even when pool events omit `backend_id` or send `IP:port`.
        let backend_id = if is_pool && sb.is_node_pool_registration(&req.medium_endpoints) {
            req.medium_endpoints
                .values()
                .next()
                .map(|url| crate::backend::lookup_ip_from_backend_id(url))
                .filter(|ip| !ip.is_empty())
                .unwrap_or_else(|| req.instance_id.clone())
        } else {
            req.instance_id.clone()
        };
        let (match_mode, use_ip_index) = subscriber_match(is_pool);
        let ip_index = if use_ip_index {
            Some(Arc::clone(&self.hbm_ip_index))
        } else {
            None
        };

        // Start replacement subscribers before acquiring the instances write
        // lock; only the test seam disables real ZMQ connections.
        let subscribers: Vec<(String, crate::zmq_subscriber::ZmqSubscriber)> =
            if self.subscriber_connections_enabled {
                let mut subs = Vec::with_capacity(endpoint_media.len());
                for (ep_url, default_media) in &endpoint_media {
                    // Engine publishers expose HBM/NPU events; CPU and Disk
                    // endpoints are pool publishers and use PoolEvent wire data.
                    let event_source = crate::zmq_subscriber::event_source_for_media(default_media);
                    let sub = crate::zmq_subscriber::ZmqSubscriber::connect(
                        ep_url.clone(),
                        req.modelname.clone(),
                        req.tenant_id.clone(),
                        Arc::clone(&self.indexer),
                        req.block_size,
                        backend_id.clone(),
                        req.dp_rank,
                        default_media.clone(),
                        event_source,
                        match_mode,
                        ip_index.clone(),
                    )?;
                    subs.push((ep_url.clone(), sub));
                }
                subs
            } else {
                Vec::new()
            };

        // Build normalized medium_endpoints for storage.
        let mut normalized_mediums: HashMap<String, String> = HashMap::new();
        for (ep_url, media) in &endpoint_media {
            for m in media {
                normalized_mediums.insert(m.as_str().to_lowercase(), ep_url.clone());
            }
        }

        self.indexer.get_or_create(&req.modelname, &req.tenant_id);

        {
            let mut instances = self.instances.write().await;
            let entry = instances
                .entry(req.instance_id.clone())
                .or_insert_with(|| WorkerEntry {
                    instance_id: req.instance_id.clone(),
                    model_name: req.modelname.clone(),
                    tenant_id: req.tenant_id.clone(),
                    block_size: req.block_size,
                    store_backend: req.store_backend.clone(),
                    store_backends: HashMap::new(),
                    endpoints: HashMap::new(),
                });

            entry.store_backend = req.store_backend.clone();
            entry
                .store_backends
                .insert(req.dp_rank, req.store_backend.clone());
            entry.endpoints.insert(
                req.dp_rank,
                EndpointInfo {
                    medium_endpoints: normalized_mediums,
                    engine_type: req.engine_type.clone(),
                    dp_rank: req.dp_rank,
                    replay_endpoint: req.replay_endpoint.clone(),
                    exact_dp_npu_provenance,
                },
            );
        }

        tracing::info!(
            instance_id = %req.instance_id, dp_rank = req.dp_rank,
            model = %req.modelname, tenant = %req.tenant_id,
            engine_type = %req.engine_type, store_backend = %req.store_backend,
            num_endpoints = endpoint_media.len(),
            "worker registered"
        );

        {
            let mut subs = self.zmq_subscribers.write().await;
            for (ep_url, sub) in subscribers {
                subs.insert((req.instance_id.clone(), req.dp_rank, ep_url), sub);
            }
        }

        if let Some(ref replay_ep) = req.replay_endpoint {
            // Exact-tuple re-registration deliberately skips replay after the
            // old NPU generation was invalidated above. Replay is not a
            // complete warm snapshot, so event-driven rebuild is safer.
            if !replay_ep.is_empty() && !tuple_exists {
                let replay_ep = replay_ep.clone();
                let modelname = req.modelname.clone();
                let tenant_id = req.tenant_id.clone();
                let block_size = req.block_size;
                let indexer = Arc::clone(&self.indexer);
                let replay_backend_id = backend_id.clone();
                let ip_index = ip_index.clone();

                // Keep replay ordered with this instance's lifecycle while
                // offloading blocking ZMQ I/O from the tokio runtime.
                let replay_result = tokio::task::spawn_blocking(move || {
                    crate::zmq_subscriber::replay_events(
                        &replay_ep,
                        &modelname,
                        &tenant_id,
                        block_size,
                        &indexer,
                        &replay_backend_id,
                        if is_pool {
                            crate::zmq_subscriber::EventSource::Pool
                        } else {
                            crate::zmq_subscriber::EventSource::Engine
                        },
                        match_mode,
                        &ip_index,
                    );
                })
                .await;
                if let Err(error) = replay_result {
                    tracing::warn!(
                        instance_id = %req.instance_id,
                        dp_rank = req.dp_rank,
                        "replay task failed: {error}"
                    );
                }
            }
        }

        Ok(())
    }

    /// Unregister a worker endpoint.
    pub async fn unregister(&self, req: &UnregisterRequest) -> Result<(), KvConductorError> {
        let lifecycle_lock = self.lifecycle_lock(&req.instance_id);
        let lifecycle_guard = lifecycle_lock.lock().await;
        let registration = {
            let instances = self.instances.read().await;
            instances.get(&req.instance_id).and_then(|entry| {
                entry.endpoints.get(&req.dp_rank).map(|endpoint| {
                    (
                        entry.model_name.clone(),
                        entry.tenant_id.clone(),
                        endpoint.medium_endpoints.clone(),
                    )
                })
            })
        };
        let Some((registered_model, registered_tenant, old_medium_endpoints)) = registration else {
            drop(lifecycle_guard);
            self.release_lifecycle_lock_if_unused(&req.instance_id, &lifecycle_lock);
            return Err(KvConductorError::InstanceNotFound {
                instance_id: req.instance_id.clone(),
            });
        };

        let old_subscribers = self.take_subscribers(&req.instance_id, req.dp_rank).await;
        Self::shutdown_subscribers(old_subscribers).await;
        remove_hbm_ip_index_entries(
            &self.hbm_ip_index,
            &old_medium_endpoints,
            &req.instance_id,
            req.dp_rank,
        );

        // Remove from indexer tree across all storage media (NPU/CPU/DISK)
        // Use the *registered* model/tenant (not the request's) to ensure
        // cleanup targets the correct indexer entry.
        let indexer_entry = self.indexer.get(&registered_model, &registered_tenant);
        if let Some(ie) = &indexer_entry {
            ie.remove_worker_all_media(&req.instance_id, req.dp_rank);
        }

        let mut instances = self.instances.write().await;
        let entry = instances
            .get_mut(&req.instance_id)
            .expect("per-instance lifecycle lock keeps registration present");
        entry.endpoints.remove(&req.dp_rank);
        entry.store_backends.remove(&req.dp_rank);

        tracing::info!(
            instance_id = %req.instance_id,
            dp_rank = req.dp_rank,
            "worker unregistered"
        );

        let removed_instance = entry.endpoints.is_empty();
        if removed_instance {
            instances.remove(&req.instance_id);
            self.indexer
                .remove_if_empty(&registered_model, &registered_tenant);
        }
        drop(instances);
        drop(lifecycle_guard);

        if removed_instance {
            self.release_lifecycle_lock_if_unused(&req.instance_id, &lifecycle_lock);
        }

        Ok(())
    }

    /// Query KV cache overlap for a token sequence.
    pub async fn query(&self, req: &QueryRequest) -> Result<QueryResponse, KvConductorError> {
        let result = self
            .indexer
            .query(&req.model, &req.tenant_id, &req.token_ids, req.block_size);
        self.finish_query(result, &req.model, &req.tenant_id, req.block_size)
            .await
    }

    /// Query KV cache overlap using pre-computed block hashes.
    pub async fn query_by_hash(
        &self,
        req: &QueryByHashRequest,
    ) -> Result<QueryResponse, KvConductorError> {
        let hashes: Vec<LocalBlockHash> = req
            .block_hashes
            .iter()
            .map(|&h| LocalBlockHash(h))
            .collect();
        let result = self
            .indexer
            .query_by_hash(&req.model, &req.tenant_id, &hashes);
        self.finish_query(result, &req.model, &req.tenant_id, req.block_size)
            .await
    }

    async fn finish_query(
        &self,
        result: Result<QueryResponse, KvConductorError>,
        model_name: &str,
        tenant_id: &str,
        block_size: u32,
    ) -> Result<QueryResponse, KvConductorError> {
        let mut response = match result {
            Ok(response) => response,
            Err(KvConductorError::NoWorkers { .. })
                if self
                    .has_registered_namespace(model_name, tenant_id, block_size)
                    .await =>
            {
                let mut response = QueryResponse::default();
                response
                    .tenants
                    .insert(tenant_id.to_string(), HashMap::new());
                response
            }
            Err(error) => return Err(error),
        };
        self.annotate_exact_dp_provenance(&mut response).await;
        Ok(response)
    }

    async fn has_registered_namespace(
        &self,
        model_name: &str,
        tenant_id: &str,
        block_size: u32,
    ) -> bool {
        self.instances.read().await.values().any(|entry| {
            entry.model_name == model_name
                && entry.tenant_id == tenant_id
                && entry.block_size == block_size
                && !entry.endpoints.is_empty()
        })
    }

    async fn annotate_exact_dp_provenance(&self, response: &mut QueryResponse) {
        let instances = self.instances.read().await;
        for (tenant_id, response_instances) in &mut response.tenants {
            for (instance_id, match_data) in response_instances {
                let Some(entry) = instances.get(instance_id) else {
                    continue;
                };
                if entry.tenant_id != *tenant_id {
                    continue;
                }
                for (dp_rank, blocks) in &mut match_data.dp {
                    let Ok(dp_rank) = dp_rank.parse::<u32>() else {
                        continue;
                    };
                    blocks.exact_dp_npu_provenance = entry
                        .endpoints
                        .get(&dp_rank)
                        .is_some_and(|endpoint| endpoint.exact_dp_npu_provenance);
                }
            }
        }
    }

    /// Apply a batch of KV cache events (engine-style, HTTP POST /events).
    ///
    /// The tuple must already be registered. This keeps NPU provenance
    /// server-owned: a client cannot seed indexer state before registration
    /// and then inherit the registration's exact-DP capability.
    ///
    /// When `model_name` / `tenant_id` are provided, they take precedence over
    /// the registered values. An unregistered tuple always returns
    /// [`KvConductorError::InstanceNotFound`] without creating indexer state.
    pub async fn apply_events(
        &self,
        instance_id: &str,
        dp_rank: u32,
        events: &[KvCacheEvent],
        model_name: Option<&str>,
        tenant_id: Option<&str>,
    ) -> Result<usize, KvConductorError> {
        // Resolve model_name / tenant_id from the registered tuple. Explicit
        // values may narrow the registered namespace but cannot create one.
        let (model_name, tenant_id) =
            {
                let instances = self.instances.read().await;
                let entry = instances.get(instance_id).ok_or_else(|| {
                    KvConductorError::InstanceNotFound {
                        instance_id: instance_id.to_string(),
                    }
                })?;
                entry.endpoints.get(&dp_rank).ok_or_else(|| {
                    KvConductorError::InstanceNotFound {
                        instance_id: instance_id.to_string(),
                    }
                })?;
                (
                    model_name.unwrap_or(&entry.model_name).to_string(),
                    tenant_id.unwrap_or(&entry.tenant_id).to_string(),
                )
            };

        let indexer_entry = self.indexer.get_or_create(&model_name, &tenant_id);

        let mut applied = 0;
        for event in events {
            let (canonical, medium_str, backend_id_opt) = event.data.normalize();

            let event_desc = match &canonical {
                KvCacheEventData::Stored(s) => format!("Stored({} blocks)", s.blocks.len()),
                KvCacheEventData::Removed { block_hashes } => {
                    format!("Removed({} hashes)", block_hashes.len())
                }
                KvCacheEventData::Cleared => "Cleared".to_string(),
            };
            tracing::debug!(
                instance_id = %instance_id,
                dp_rank,
                event_type = %event_desc,
                medium = ?medium_str,
                backend_id = ?backend_id_opt,
                "normalized event"
            );

            let medium = medium_str
                .as_deref()
                .map(StorageMedium::parse)
                .unwrap_or(StorageMedium::Npu);

            if medium == StorageMedium::Npu {
                let mut instances = self.instances.write().await;
                if let Some(endpoint) = instances
                    .get_mut(instance_id)
                    .and_then(|entry| entry.endpoints.get_mut(&dp_rank))
                {
                    endpoint.exact_dp_npu_provenance = false;
                }
            }

            let backend_id = backend_id_opt.as_deref().unwrap_or(instance_id).to_string();

            let worker = WorkerKey {
                instance_id: instance_id.to_string(),
                backend_id,
                dp_rank,
                medium,
            };

            indexer_entry.apply_event(&worker, &canonical)?;
            applied += 1;
        }

        Ok(applied)
    }

    /// List all registered workers (debug endpoint).
    pub async fn list_workers(&self) -> Vec<WorkerSummary> {
        let instances = self.instances.read().await;
        instances
            .values()
            .map(|entry| WorkerSummary {
                instance_id: entry.instance_id.clone(),
                model_name: entry.model_name.clone(),
                tenant_id: entry.tenant_id.clone(),
                block_size: entry.block_size,
                endpoints: entry.endpoints.clone(),
            })
            .collect()
    }

    /// Get indexer summary (debug endpoint).
    pub fn indexer_summary(&self) -> Vec<crate::indexer::IndexerSummary> {
        self.indexer.summary()
    }

    /// Access the underlying indexer (for advanced use).
    pub fn indexer(&self) -> &Arc<Indexer> {
        &self.indexer
    }

    /// Sweep stale caches and reclaim indexers which have no active registration.
    pub async fn maintenance(&self) -> usize {
        // Keep the read guard through the pass so registration cannot race
        // between the protected-key snapshot and empty-entry reclamation.
        let instances = self.instances.read().await;
        let protected = instances
            .values()
            .map(|entry| IndexerKey {
                model_name: entry.model_name.clone(),
                tenant_id: entry.tenant_id.clone(),
            })
            .collect();
        self.indexer.maintenance(&protected)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerSummary {
    pub instance_id: String,
    pub model_name: String,
    pub tenant_id: String,
    pub block_size: u32,
    pub endpoints: HashMap<u32, EndpointInfo>,
}

use serde::Serialize;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    fn register_request(
        instance_id: &str,
        dp_rank: u32,
        ip: Option<&str>,
        store_backend: &str,
    ) -> RegisterRequest {
        RegisterRequest {
            instance_id: instance_id.to_string(),
            medium_endpoints: ip
                .map(|ip| HashMap::from([("npu".to_string(), format!("tcp://{ip}:50090"))]))
                .unwrap_or_default(),
            endpoint: None,
            engine_type: "vllm".to_string(),
            modelname: "test-model".to_string(),
            block_size: 4,
            dp_rank,
            store_backend: store_backend.to_string(),
            replay_endpoint: None,
            tenant_id: "default".to_string(),
        }
    }

    fn unregister_request(instance_id: &str, dp_rank: u32) -> UnregisterRequest {
        UnregisterRequest {
            instance_id: instance_id.to_string(),
            engine_type: "vllm".to_string(),
            modelname: "test-model".to_string(),
            block_size: 4,
            dp_rank,
            tenant_id: "default".to_string(),
        }
    }

    fn seed_worker(
        registry: &WorkerRegistry,
        instance_id: &str,
        dp_rank: u32,
        medium: StorageMedium,
        tokens_hash: u64,
    ) {
        let entry = registry.indexer.get_or_create("test-model", "default");
        let worker = WorkerKey {
            instance_id: instance_id.to_string(),
            backend_id: instance_id.to_string(),
            dp_rank,
            medium,
        };
        entry
            .apply_event(
                &worker,
                &KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: 1000 + u64::from(dp_rank),
                        tokens_hash,
                    }],
                }),
            )
            .unwrap();
    }

    async fn reported_npu_blocks(
        registry: &WorkerRegistry,
        instance_id: &str,
        dp_rank: u32,
        tokens_hash: u64,
    ) -> u32 {
        registry
            .query_by_hash(&QueryByHashRequest {
                model: "test-model".to_string(),
                block_size: 4,
                block_hashes: vec![tokens_hash],
                tenant_id: "default".to_string(),
            })
            .await
            .unwrap()
            .tenants
            .get("default")
            .and_then(|instances| instances.get(instance_id))
            .and_then(|instance| instance.dp.get(&dp_rank.to_string()))
            .map(|dp| dp.npu_blocks)
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn test_node_pool_replay_without_backend_id_clears_node_dps() {
        let registry = WorkerRegistry::new();
        let entry = registry.indexer.get_or_create("model", "default");
        for dp_rank in [0, 1] {
            let request: RegisterRequest = serde_json::from_value(serde_json::json!({
                "instance_id": "engine", "dp_rank": dp_rank, "type": "vLLM",
                "store_backend": "YuanRong", "modelname": "model", "block_size": 4,
                "medium_endpoints": {"npu": format!("tcp://127.0.0.1:{}", 15557 + dp_rank)}
            }))
            .unwrap();
            registry.register(&request).await.unwrap();
            entry
                .apply_event(
                    &WorkerKey {
                        instance_id: "engine".into(),
                        backend_id: "engine".into(),
                        dp_rank,
                        medium: StorageMedium::Cpu,
                    },
                    &KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None,
                        start_position: None,
                        blocks: vec![KvCacheStoredBlockData {
                            block_hash: 100,
                            tokens_hash: 42,
                        }],
                    }),
                )
                .unwrap();
        }
        assert!(registry
            .indexer
            .query_by_hash("model", "default", &[LocalBlockHash(42)])
            .unwrap()
            .tenants["default"]
            .contains_key("engine"));

        // A real replay ROUTER exercises registration -> replay -> pool routing.
        let context = zmq::Context::new();
        let router = context.socket(zmq::ROUTER).unwrap();
        router.set_rcvtimeo(5000).unwrap();
        router.set_sndtimeo(5000).unwrap();
        router.set_linger(0).unwrap();
        router.bind("tcp://127.0.0.1:*").unwrap();
        let replay_endpoint = router.get_last_endpoint().unwrap().unwrap();
        let publisher = tokio::task::spawn_blocking(move || {
            let request = router.recv_multipart(0).unwrap();
            let payload = rmp_serde::to_vec(&(
                0i64,
                vec![serde_json::json!({
                    "event_type": "cleared", "medium": "cpu"
                })],
                0u32,
            ))
            .unwrap();
            for (seq, payload) in [(0u64, payload), (u64::MAX, Vec::new())] {
                router
                    .send_multipart(
                        [
                            request[0].as_slice(),
                            &[],
                            &seq.to_be_bytes(),
                            payload.as_slice(),
                        ],
                        0,
                    )
                    .unwrap();
            }
        });
        let request: RegisterRequest = serde_json::from_value(serde_json::json!({
            "instance_id": "yuanrong-pool-127.0.0.1-model", "dp_rank": 0,
            "type": "vLLM", "store_backend": "YuanRong", "modelname": "model",
            "block_size": 4, "medium_endpoints": {"cpu": "tcp://127.0.0.1:25558"},
            "replay_endpoint": replay_endpoint
        }))
        .unwrap();
        registry.register(&request).await.unwrap();
        publisher.await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                match registry
                    .indexer
                    .query_by_hash("model", "default", &[LocalBlockHash(42)])
                {
                    Err(KvConductorError::NoWorkers { .. }) => break,
                    Ok(response) if !response.tenants["default"].contains_key("engine") => break,
                    Ok(_) => {}
                    Err(error) => panic!("unexpected query failure: {error}"),
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("pool replay must clear CPU blocks on every DP on the node");
    }

    #[test]
    fn test_extract_ip_ipv4() {
        assert_eq!(
            extract_ip_from_endpoint("tcp://10.0.0.1:5557"),
            Some("10.0.0.1".into())
        );
    }

    #[test]
    fn test_extract_ip_ipv6_bracketed() {
        assert_eq!(
            extract_ip_from_endpoint("tcp://[::1]:5557"),
            Some("::1".into())
        );
        assert_eq!(
            extract_ip_from_endpoint("tcp://[2001:db8::1]:15557"),
            Some("2001:db8::1".into())
        );
    }

    #[test]
    fn test_extract_ip_hostname() {
        assert_eq!(
            extract_ip_from_endpoint("tcp://kv-conductor:13333"),
            Some("kv-conductor".into())
        );
    }

    #[test]
    fn test_extract_ip_invalid() {
        assert_eq!(extract_ip_from_endpoint("invalid"), None);
        assert_eq!(extract_ip_from_endpoint("tcp://"), None);
        assert_eq!(extract_ip_from_endpoint("tcp://[:5557"), None);
    }

    #[tokio::test]
    async fn test_1000_identical_reregistrations_keep_one_tuple_and_ip_mapping() {
        let registry = WorkerRegistry::without_subscriber_connections();
        let request = register_request("prefill-0", 0, Some("10.0.0.1"), "Mooncake");

        for _ in 0..1000 {
            registry.register(&request).await.unwrap();
        }

        let workers = registry.list_workers().await;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].endpoints.len(), 1);
        assert_eq!(
            registry.hbm_ip_index.read().get("10.0.0.1"),
            Some(&vec![("prefill-0".to_string(), 0)])
        );
    }

    #[tokio::test]
    async fn test_exact_dp_npu_provenance_is_server_owned_per_tuple() {
        let registry = WorkerRegistry::without_subscriber_connections();
        let direct = register_request("prefill", 0, Some("10.0.0.1"), "Mooncake");
        let mut ip_only_pool = register_request("mooncake-pool", 0, None, "Mooncake");
        ip_only_pool.endpoint = Some("tcp://pool:5557".to_string());
        registry.register(&direct).await.unwrap();
        registry.register(&ip_only_pool).await.unwrap();

        let workers = registry.list_workers().await;
        let direct = workers
            .iter()
            .find(|worker| worker.instance_id == "prefill")
            .unwrap();
        let pool = workers
            .iter()
            .find(|worker| worker.instance_id == "mooncake-pool")
            .unwrap();

        assert!(direct.endpoints[&0].exact_dp_npu_provenance);
        assert!(!pool.endpoints[&0].exact_dp_npu_provenance);
    }

    #[tokio::test]
    async fn test_sparse_zero_query_and_provenance_annotation() {
        let registry = WorkerRegistry::without_subscriber_connections();
        registry
            .register(&register_request(
                "prefill",
                0,
                Some("10.0.0.1"),
                "Mooncake",
            ))
            .await
            .unwrap();

        let empty = registry
            .query_by_hash(&QueryByHashRequest {
                model: "test-model".to_string(),
                block_size: 4,
                block_hashes: vec![123],
                tenant_id: "default".to_string(),
            })
            .await
            .unwrap();
        assert!(empty.tenants["default"].is_empty());

        seed_worker(&registry, "prefill", 0, StorageMedium::Npu, 123);
        let hit = registry
            .query_by_hash(&QueryByHashRequest {
                model: "test-model".to_string(),
                block_size: 4,
                block_hashes: vec![123],
                tenant_id: "default".to_string(),
            })
            .await
            .unwrap();
        assert!(hit.tenants["default"]["prefill"].dp["0"].exact_dp_npu_provenance);
    }

    #[tokio::test]
    async fn test_http_npu_event_taints_exact_dp_provenance() {
        let registry = WorkerRegistry::without_subscriber_connections();
        registry
            .register(&register_request(
                "prefill",
                0,
                Some("10.0.0.1"),
                "Mooncake",
            ))
            .await
            .unwrap();

        let event = KvCacheEvent {
            event_id: 1,
            data: KvEventWirePayload {
                event_type: "stored".to_string(),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 99,
                    tokens_hash: 123,
                }],
                medium: Some("npu".to_string()),
                ..Default::default()
            },
            dp_rank: 0,
        };
        registry
            .apply_events("prefill", 0, &[event], None, None)
            .await
            .unwrap();

        assert!(!registry.list_workers().await[0].endpoints[&0].exact_dp_npu_provenance);
        let response = registry
            .query_by_hash(&QueryByHashRequest {
                model: "test-model".to_string(),
                block_size: 4,
                block_hashes: vec![123],
                tenant_id: "default".to_string(),
            })
            .await
            .unwrap();
        assert!(!response.tenants["default"]["prefill"].dp["0"].exact_dp_npu_provenance);
    }

    #[tokio::test]
    async fn test_unregistered_http_events_cannot_seed_exact_dp_state() {
        let registry = WorkerRegistry::without_subscriber_connections();
        let event = KvCacheEvent {
            event_id: 1,
            data: KvEventWirePayload {
                event_type: "stored".to_string(),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 99,
                    tokens_hash: 123,
                }],
                ..Default::default()
            },
            dp_rank: 0,
        };

        let error = registry
            .apply_events("prefill", 0, &[event], Some("test-model"), None)
            .await
            .unwrap_err();

        assert!(matches!(error, KvConductorError::InstanceNotFound { .. }));
        assert!(registry.indexer.get("test-model", "default").is_none());
    }

    #[tokio::test]
    async fn test_http_events_require_the_registered_dp_rank() {
        let registry = WorkerRegistry::without_subscriber_connections();
        registry
            .register(&register_request(
                "prefill",
                0,
                Some("10.0.0.1"),
                "Mooncake",
            ))
            .await
            .unwrap();
        let event = KvCacheEvent {
            event_id: 1,
            data: KvEventWirePayload {
                event_type: "stored".to_string(),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: 99,
                    tokens_hash: 123,
                }],
                ..Default::default()
            },
            dp_rank: 1,
        };

        let error = registry
            .apply_events("prefill", 1, &[event], Some("test-model"), None)
            .await
            .unwrap_err();

        assert!(matches!(error, KvConductorError::InstanceNotFound { .. }));
        assert!(registry
            .indexer
            .get("test-model", "default")
            .unwrap()
            .worker_keys()
            .is_empty());
    }

    #[test]
    fn test_live_and_replay_match_context_is_exact_for_engine_and_ip_only_for_pool() {
        let (engine_mode, engine_uses_ip_index) = subscriber_match(false);
        assert_eq!(engine_mode, MatchMode::None);
        assert!(!engine_uses_ip_index);

        let (pool_mode, pool_uses_ip_index) = subscriber_match(true);
        assert_eq!(pool_mode, MatchMode::IpOnly);
        assert!(pool_uses_ip_index);
    }

    #[tokio::test]
    async fn test_reregister_clears_only_exact_dp_hbm() {
        let registry = WorkerRegistry::without_subscriber_connections();
        let dp0 = register_request("prefill", 0, Some("10.0.0.1"), "Mooncake");
        let dp1 = register_request("prefill", 1, Some("10.0.0.1"), "Mooncake");
        registry.register(&dp0).await.unwrap();
        registry.register(&dp1).await.unwrap();
        seed_worker(&registry, "prefill", 0, StorageMedium::Npu, 900);
        seed_worker(&registry, "prefill", 1, StorageMedium::Npu, 900);

        registry.register(&dp0).await.unwrap();

        assert_eq!(reported_npu_blocks(&registry, "prefill", 0, 900).await, 0);
        assert_eq!(reported_npu_blocks(&registry, "prefill", 1, 900).await, 1);
    }

    #[tokio::test]
    async fn test_backend_change_is_scoped_to_exact_dp() {
        let registry = WorkerRegistry::without_subscriber_connections();
        let mut dp0 = register_request("prefill", 0, Some("10.0.0.1"), "Mooncake");
        let dp1 = register_request("prefill", 1, Some("10.0.0.1"), "Mooncake");
        registry.register(&dp0).await.unwrap();
        registry.register(&dp1).await.unwrap();
        seed_worker(&registry, "prefill", 1, StorageMedium::Cpu, 901);

        dp0.store_backend = "YuanRong".to_string();
        registry.register(&dp0).await.unwrap();
        registry.register(&dp1).await.unwrap();

        let entry = registry.indexer.get("test-model", "default").unwrap();
        assert!(entry.worker_keys().iter().any(|worker| {
            worker.instance_id == "prefill"
                && worker.dp_rank == 1
                && worker.medium == StorageMedium::Cpu
        }));
    }

    #[tokio::test]
    async fn test_identity_conflict_is_typed_and_state_is_unchanged() {
        let registry = WorkerRegistry::without_subscriber_connections();
        let request = register_request("prefill", 0, Some("10.0.0.1"), "Mooncake");
        registry.register(&request).await.unwrap();
        seed_worker(&registry, "prefill", 0, StorageMedium::Npu, 902);

        let mut model_conflict = request.clone();
        model_conflict.modelname = "other-model".to_string();
        model_conflict
            .medium_endpoints
            .insert("npu".to_string(), "tcp://10.0.0.2:50090".to_string());
        let mut tenant_conflict = request.clone();
        tenant_conflict.tenant_id = "other-tenant".to_string();
        let mut block_size_conflict = request.clone();
        block_size_conflict.block_size = 256;

        for conflict in [model_conflict, tenant_conflict, block_size_conflict] {
            let error = registry.register(&conflict).await.unwrap_err();
            assert!(matches!(
                error,
                KvConductorError::RegistrationIdentityConflict { .. }
            ));
        }
        assert_eq!(registry.list_workers().await[0].model_name, "test-model");
        assert_eq!(
            registry.hbm_ip_index.read().get("10.0.0.1"),
            Some(&vec![("prefill".to_string(), 0)])
        );
        assert!(!registry.hbm_ip_index.read().contains_key("10.0.0.2"));
        assert_eq!(reported_npu_blocks(&registry, "prefill", 0, 902).await, 1);
    }

    #[tokio::test]
    async fn test_unregister_unknown_tuple_is_noop_and_releases_lifecycle_lock() {
        let registry = WorkerRegistry::without_subscriber_connections();
        registry
            .register(&register_request(
                "prefill",
                0,
                Some("10.0.0.1"),
                "Mooncake",
            ))
            .await
            .unwrap();
        let before_ip_index = registry.hbm_ip_index.read().clone();

        let error = registry
            .unregister(&unregister_request("unknown", 7))
            .await
            .unwrap_err();

        assert!(matches!(error, KvConductorError::InstanceNotFound { .. }));
        let workers = registry.list_workers().await;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].instance_id, "prefill");
        assert!(workers[0].endpoints.contains_key(&0));
        assert_eq!(*registry.hbm_ip_index.read(), before_ip_index);
        assert!(!registry.lifecycle_locks.contains_key("unknown"));
    }

    #[tokio::test]
    async fn test_10000_randomized_registry_lifecycle_events_match_reference_state() {
        let registry = WorkerRegistry::without_subscriber_connections();
        let mut active: HashMap<(String, u32), String> = HashMap::new();
        let mut random = 0xCFF8_7175_D15C_A11Eu64;

        for _ in 0..10_000 {
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let instance_id = format!("prefill-{}", (random >> 8) % 8);
            let dp_rank = ((random >> 16) % 4) as u32;
            let key = (instance_id.clone(), dp_rank);

            if random.is_multiple_of(4) {
                let result = registry
                    .unregister(&unregister_request(&instance_id, dp_rank))
                    .await;
                if active.remove(&key).is_some() {
                    result.unwrap();
                } else {
                    assert!(matches!(
                        result,
                        Err(KvConductorError::InstanceNotFound { .. })
                    ));
                }
            } else {
                let ip = format!("10.0.{}.{}", (random >> 24) % 4, (random >> 32) % 8 + 1);
                registry
                    .register(&register_request(
                        &instance_id,
                        dp_rank,
                        Some(&ip),
                        "Mooncake",
                    ))
                    .await
                    .unwrap();
                active.insert(key, ip);
            }

            let actual_tuples: HashSet<(String, u32)> = registry
                .list_workers()
                .await
                .into_iter()
                .flat_map(|worker| {
                    worker
                        .endpoints
                        .into_keys()
                        .map(move |dp| (worker.instance_id.clone(), dp))
                })
                .collect();
            assert_eq!(actual_tuples, active.keys().cloned().collect());

            let mut expected_ip_index: HashMap<String, HashSet<(String, u32)>> = HashMap::new();
            for (worker, ip) in &active {
                expected_ip_index
                    .entry(ip.clone())
                    .or_default()
                    .insert(worker.clone());
            }
            let actual_ip_index = registry.hbm_ip_index.read();
            assert_eq!(actual_ip_index.len(), expected_ip_index.len());
            for (ip, workers) in actual_ip_index.iter() {
                let unique: HashSet<(String, u32)> = workers.iter().cloned().collect();
                assert_eq!(
                    unique.len(),
                    workers.len(),
                    "duplicate IP index entry for {ip}"
                );
                assert_eq!(Some(&unique), expected_ip_index.get(ip));
            }
        }

        let remaining: Vec<(String, u32)> = active.into_keys().collect();
        for (instance_id, dp_rank) in remaining {
            registry
                .unregister(&unregister_request(&instance_id, dp_rank))
                .await
                .unwrap();
        }
        assert!(registry.list_workers().await.is_empty());
        assert!(registry.hbm_ip_index.read().is_empty());
        assert!(registry.indexer_summary().is_empty());
        assert!(registry.lifecycle_locks.is_empty());
    }

    #[tokio::test]
    async fn test_unrelated_registration_is_not_blocked_by_subscriber_shutdown() {
        let registry = Arc::new(WorkerRegistry::without_subscriber_connections());
        registry
            .register(&register_request("blocked", 0, None, "Mooncake"))
            .await
            .unwrap();
        let (subscriber, cancel, stopped) =
            crate::zmq_subscriber::ZmqSubscriber::pending_for_test();
        registry.zmq_subscribers.write().await.insert(
            ("blocked".to_string(), 0, "test://pending".to_string()),
            subscriber,
        );

        let blocked_registry = Arc::clone(&registry);
        let unregister = tokio::spawn(async move {
            blocked_registry
                .unregister(&unregister_request("blocked", 0))
                .await
        });
        cancel.cancelled().await;

        let unrelated = tokio::time::timeout(
            Duration::from_millis(100),
            registry.register(&register_request("unrelated", 0, None, "Mooncake")),
        )
        .await;
        stopped.store(true, Ordering::Release);
        unregister.await.unwrap().unwrap();

        assert!(
            matches!(unrelated, Ok(Ok(()))),
            "unrelated registration waited for another worker's shutdown"
        );
    }

    #[tokio::test]
    async fn test_replay_does_not_hold_instances_write_lock() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (accepted_tx, accepted_rx) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            if let Ok((_stream, _)) = listener.accept() {
                let _ = accepted_tx.send(());
                std::thread::sleep(Duration::from_secs(3));
            }
        });

        let registry = Arc::new(WorkerRegistry::without_subscriber_connections());
        let mut request = register_request("replay", 0, Some("10.0.0.1"), "Mooncake");
        request.replay_endpoint = Some(format!("tcp://127.0.0.1:{port}"));
        let registration_registry = Arc::clone(&registry);
        let registration =
            tokio::spawn(async move { registration_registry.register(&request).await });

        tokio::time::timeout(Duration::from_secs(1), accepted_rx)
            .await
            .expect("replay did not connect")
            .expect("replay accept signal dropped");

        let workers =
            tokio::time::timeout(Duration::from_millis(200), registry.list_workers()).await;
        assert!(
            workers.is_ok(),
            "list_workers waited for replay while holding the global instances lock"
        );
        registration.await.unwrap().unwrap();
    }
}
