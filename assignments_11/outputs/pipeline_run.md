## Reflection

The pipeline failed on the first run. There was a bug in Transform task:
`len(max_records)` instead of `max_records`. `max_records` is an integer and does not have a length.

The second run was successful - dashboard showed all the three tasks - Extract, Transform, and Load - as Completed, with no retries.

To run on a daily schedule, I'd create a deployment. And also I would only load data for the next 24 hours instead of 7 days, since it is run daily.
