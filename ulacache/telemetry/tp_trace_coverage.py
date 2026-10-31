"""Validate actual TP fused-begin headers without claiming an empty pin bucket."""


def validate_fused_begin_rotation(observations, recent_count, valid_queries):
    """After native begin, expiry and current-row pins have both completed.

    The old header/member count identifies actual expiration work. New current
    members and cold removals can be current pins, so the DPA empty-bucket and
    exact cold-delta checks do not apply at this later observation boundary.
    """
    if type(recent_count) is not int or recent_count < 1:
        raise ValueError("A positive native recent-bucket count is required")
    rows, limits = list(observations), list(valid_queries)
    if not rows or len(rows) != len(limits):
        raise ValueError("Every observed begin needs its valid query bound")
    previous = None
    visited = set()
    headers = members = nonempty = 0
    for row, queries in zip(rows, limits):
        if len(row) != 8 or any(type(value) is not int for value in row):
            raise ValueError("Invalid native fused-begin observation")
        if type(queries) is not int or queries < 0:
            raise ValueError("Invalid native current query bound")
        (
            epoch,
            bucket,
            age,
            old_count,
            new_age,
            current_count,
            cold_before,
            cold_after,
        ) = row
        if epoch < 1 or bucket != epoch % recent_count or age < -1 or age > epoch:
            raise ValueError("Invalid native group clock or bucket identity")
        if previous is not None and epoch != previous + 1:
            raise ValueError("A native group clock skipped or repeated")
        if min(old_count, current_count, cold_before, cold_after) < 0:
            raise ValueError("Negative native bucket membership")
        expired = age >= 0 and epoch - age >= recent_count
        if new_age != epoch or current_count > queries:
            raise ValueError("Fused begin has an invalid current bucket")
        if old_count and not expired:
            raise ValueError("A nonempty reused bucket was not old enough to expire")
        moved = old_count if expired else 0
        if not -queries <= cold_after - cold_before <= moved:
            raise ValueError(
                "Cold membership cannot be explained by expiry and current pins"
            )
        headers += int(expired)
        members += moved
        nonempty += int(expired and old_count > 0)
        visited.add(bucket)
        previous = epoch
    if len(visited) != recent_count:
        raise ValueError("The native group did not cover a full bucket rotation")
    return dict(
        target_forwards=len(rows),
        first_epoch=rows[0][0],
        last_epoch=rows[-1][0],
        native_recent_buckets=recent_count,
        visited_buckets=len(visited),
        expired_headers=headers,
        expired_members=members,
        nonempty_expirations=nonempty,
        full_rotation=True,
        boundary="Fused native expiry plus current pins",
        pure_expiry_time_measured=False,
    )
