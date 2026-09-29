-- Truck leaves soon but a few totes are still in the building: suggest holding it 10 minutes.
SELECT 'delay_truck'                        AS rec_type,
       'truck:' || t.truck_id               AS rec_key,
       t.truck_id                           AS target,
       t.truck_id                           AS truck_id,
       t.seconds_to_departure               AS seconds_to_departure,
       3                                    AS severity
FROM trucks t
WHERE t.seconds_to_departure BETWEEN 0 AND 300
  AND t.totes_missing BETWEEN 1 AND 12
  AND t.delay_s = 0
