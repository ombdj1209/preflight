-- A tote is waiting for a missing item and replenishment looks too late for its truck.
SELECT 'cancel_item'                        AS rec_type,
       'short:' || sh.order_id              AS rec_key,
       sh.order_id                          AS target,
       sh.truck_id                          AS truck_id,
       sh.seconds_to_departure              AS seconds_to_departure,
       CASE WHEN sh.seconds_to_departure < 600 THEN 3 ELSE 2 END AS severity
FROM shortages sh, system s
WHERE sh.seconds_to_departure < 1800
  AND sh.replenish_eta_s > sh.seconds_to_departure - s.conveyor_transit_s - 300
