-- Recommend releasing the most urgent pending batch early when stations have headroom.
-- Default automation would release it at due_at; accepting pulls work forward.
SELECT 'release_batch'                      AS rec_type,
       'batch:' || b.batch_id               AS rec_key,
       b.batch_id                           AS target,
       b.truck_id                           AS truck_id,
       b.seconds_to_departure               AS seconds_to_departure,
       CASE WHEN b.seconds_to_departure < 3000 THEN 3
            WHEN b.seconds_to_departure < 3600 THEN 2 ELSE 1 END AS severity
FROM pending_batches b, system s
WHERE s.queue_len < s.n_stations * 4
  AND b.seconds_to_departure < 4200
QUALIFY row_number() OVER (ORDER BY b.seconds_to_departure, b.batch_id) = 1
