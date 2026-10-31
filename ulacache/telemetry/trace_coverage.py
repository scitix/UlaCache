"""Dependency-free validation of native PB bucket-rotation observations."""


def validate_bucket_rotation(observations, recent_count):
    """Validate one active group's consecutive before/after expiry records.

    Each record is [next_epoch, bucket, old_age, old_members, new_age,
    new_members, cold_before, cold_after]. Counts describe actual native
    bucket membership, rather than estimated work from global decode steps.
    """
    if type(recent_count) is not int or recent_count < 1:
        raise ValueError("A positive native recent-bucket count is required")
    rows = list(observations)
    if not rows:
        raise ValueError("An active group has no bucket observations")
    previous = None
    expired_headers = expired_members = nonempty_expirations = 0
    visited = set()
    for row in rows:
        if len(row) != 8 or any(type(value) is not int for value in row):
            raise ValueError("Invalid native expiry observation")
        epoch, bucket, age, members, new_age, remaining, cold_before, cold_after = row
        if epoch < 1 or bucket != epoch % recent_count or age < -1:
            raise ValueError("Native group clock or bucket identity is invalid")
        if previous is not None and epoch != previous + 1:
            raise ValueError(
                "A group clock skipped or repeated within the trace window"
            )
        if min(members, remaining, cold_before, cold_after) < 0 or age > epoch:
            raise ValueError("Native bucket counts or age labels are invalid")
        expired = age >= 0 and epoch - age >= recent_count
        if new_age != epoch or remaining != 0:
            raise ValueError("Native expiry did not produce an empty current bucket")
        if age < 0 and members:
            raise ValueError("An unused bucket contains resident members")
        if not expired and members:
            raise ValueError("A reused recent bucket is not old enough to expire")
        if cold_after - cold_before != (members if expired else 0):
            raise ValueError(
                "Observed cold membership does not match actual expiration"
            )
        expired_headers += int(expired)
        expired_members += members if expired else 0
        nonempty_expirations += int(expired and members > 0)
        visited.add(bucket)
        previous = epoch
    if len(visited) != recent_count:
        raise ValueError(
            "The active group did not complete a full native bucket rotation"
        )
    return dict(
        target_forwards=len(rows),
        native_recent_buckets=recent_count,
        first_epoch=rows[0][0],
        last_epoch=rows[-1][0],
        visited_buckets=len(visited),
        expired_headers=expired_headers,
        expired_members=expired_members,
        nonempty_expirations=nonempty_expirations,
        full_rotation=True,
    )
