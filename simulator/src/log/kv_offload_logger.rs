//! Per-worker `kv_offload_event` stream: host (CPU) KV tier operations.
//!
//! The host tier (`worker/kv/shared/host_offload.rs`) is the only ledger; the
//! KV owner records each operation it performed:
//!
//! - `activate`: an admitted session request's prefix resolution, split into
//!   the GPU hit and the deeper part loaded from the host (with the H2D copy
//!   that loads it), so hits can be attributed to a tier;
//! - `store`: one queued D2H copy;
//! - `evict`: blocks the host LRU gave up.
//!
//! The stream exists only for workers with a host tier. Rows are buffered on
//! the sim thread and written once when the logger closes.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::{Float64Array, RecordBatch, StringArray, UInt16Array, UInt32Array, UInt64Array};

use crate::common::{RequestId, Time, WorkerId};
use crate::log::cost_logger::cost_artifact_stem;
use crate::log::parquet_writer::StreamingParquetWriter;
use crate::log::schemas::kv_offload_event_schema;

/// One host-tier operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvOffloadOperation {
    Activate,
    Store,
    Evict,
}

impl KvOffloadOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Activate => "activate",
            Self::Store => "store",
            Self::Evict => "evict",
        }
    }
}

/// Facts of one operation, supplied by the KV owner.
#[derive(Clone, Copy, Debug)]
pub struct KvOffloadEvent {
    pub partition_id: u16,
    pub time: Time,
    pub operation: KvOffloadOperation,
    /// The request that triggered the operation.
    pub request_id: RequestId,
    pub session_id: u32,
    /// `activate` only: prefix tokens resident on the GPU.
    pub gpu_hit_tokens: u64,
    /// `activate` only: total resident prefix when the host served a deeper
    /// hit, else 0.
    pub host_hit_tokens: u64,
    /// Token equivalents copied (`activate` load, `store`) or freed (`evict`).
    pub tokens: u64,
    /// Bytes copied per rank.
    pub bytes: u64,
    pub transfer_start: Time,
    pub transfer_done: Time,
    /// Host occupancy after the operation, token equivalents.
    pub host_used_tokens: u64,
}

struct KvOffloadEventEntry {
    sequence: u64,
    event: KvOffloadEvent,
}

/// Sim-thread buffer for one worker's host-tier operations.
pub struct KvOffloadLogger {
    path: std::path::PathBuf,
    pool_tag: &'static str,
    worker_id: WorkerId,
    rows: Vec<KvOffloadEventEntry>,
    closed: bool,
}

impl KvOffloadLogger {
    pub fn open_opt(
        log_dir: Option<&Path>,
        pool_tag: &'static str,
        worker_id: WorkerId,
    ) -> Option<Self> {
        let directory = log_dir?.join("raw").join("kv_offload_event");
        if let Err(error) = std::fs::create_dir_all(&directory) {
            tracing::warn!("kv_offload_event disabled: {error:#}");
            return None;
        }
        Some(Self {
            path: directory.join(format!(
                "{}.parquet",
                cost_artifact_stem(pool_tag, worker_id)
            )),
            pool_tag,
            worker_id,
            rows: Vec::new(),
            closed: false,
        })
    }

    pub fn record(&mut self, event: KvOffloadEvent) {
        if self.closed {
            return;
        }
        let sequence = self.rows.len() as u64;
        self.rows.push(KvOffloadEventEntry { sequence, event });
    }

    pub fn flush_all(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        let mut writer = StreamingParquetWriter::new(self.path.clone(), kv_offload_event_schema());
        for chunk in self.rows.chunks(8_192) {
            writer.write(&self.to_record_batch(chunk)?)?;
        }
        writer.close()?;
        Ok(())
    }

    fn to_record_batch(&self, rows: &[KvOffloadEventEntry]) -> Result<RecordBatch> {
        let column_u64 = |value: fn(&KvOffloadEvent) -> u64| {
            Arc::new(UInt64Array::from(
                rows.iter().map(|row| value(&row.event)).collect::<Vec<_>>(),
            ))
        };
        let column_ms = |value: fn(&KvOffloadEvent) -> Time| {
            Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| value(&row.event).as_ms())
                    .collect::<Vec<_>>(),
            ))
        };
        Ok(RecordBatch::try_new(
            kv_offload_event_schema(),
            vec![
                Arc::new(StringArray::from(vec![self.pool_tag; rows.len()])),
                Arc::new(UInt16Array::from(vec![self.worker_id.0; rows.len()])),
                Arc::new(UInt16Array::from(
                    rows.iter()
                        .map(|row| row.event.partition_id)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(UInt64Array::from(
                    rows.iter().map(|row| row.sequence).collect::<Vec<_>>(),
                )),
                column_ms(|event| event.time),
                Arc::new(StringArray::from(
                    rows.iter()
                        .map(|row| row.event.operation.as_str())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(UInt32Array::from(
                    rows.iter()
                        .map(|row| row.event.request_id.0)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(UInt32Array::from(
                    rows.iter()
                        .map(|row| row.event.session_id)
                        .collect::<Vec<_>>(),
                )),
                column_u64(|event| event.gpu_hit_tokens),
                column_u64(|event| event.host_hit_tokens),
                column_u64(|event| event.tokens),
                column_u64(|event| event.bytes),
                column_ms(|event| event.transfer_start),
                column_ms(|event| event.transfer_done),
                column_u64(|event| event.host_used_tokens),
            ],
        )?)
    }
}

impl Drop for KvOffloadLogger {
    fn drop(&mut self) {
        if let Err(error) = self.flush_all() {
            tracing::warn!("kv_offload_event write failed: {error:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;

    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    #[test]
    fn writes_one_row_per_recorded_operation() {
        let directory = tempdir().unwrap();
        let mut logger =
            KvOffloadLogger::open_opt(Some(directory.path()), "main", WorkerId(2)).unwrap();
        for operation in [KvOffloadOperation::Store, KvOffloadOperation::Activate] {
            logger.record(KvOffloadEvent {
                partition_id: 0,
                time: Time::from_ms(3.0),
                operation,
                request_id: RequestId(5),
                session_id: 9,
                gpu_hit_tokens: 0,
                host_hit_tokens: 2_304,
                tokens: 2_304,
                bytes: 4_096,
                transfer_start: Time::from_ms(3.0),
                transfer_done: Time::from_ms(3.5),
                host_used_tokens: 10_000,
            });
        }
        logger.flush_all().unwrap();
        let path = directory
            .path()
            .join("raw/kv_offload_event/worker_main_2.parquet");
        let batch = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(batch.num_rows(), 2);
        let operation = batch
            .column_by_name("operation")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(operation.value(1), "activate");
    }
}
