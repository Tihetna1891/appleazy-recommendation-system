from typing import List

def filter_jobs(user: dict, jobs: List[dict]) -> List[dict]:
    """
    Filters jobs based on the user's preferences:
    ✓ location (comma-separated string)
    ✓ min desired pay (int)
    ✓ job type (comma-separated: e.g., "full-time,contract")
    ✓ job setting (comma-separated: e.g., "remote,onsite")
    ✓ job status must be 'open'
    """
    user_locations = [loc.strip().lower() for loc in user.get("location", "").split(",") if loc.strip()]
    min_pay = int(user.get("minDesiredPay", 0))

    preferred_types = [t.strip().lower() for t in user.get("jobType", "").split(",") if t.strip()]
    preferred_settings = [s.strip().lower() for s in user.get("jobSetting", "").split(",") if s.strip()]

    filtered = []
    for job in jobs:
        if job.get("status", "").lower() != "open":
            continue

        job_location = job.get("location", "").strip().lower()
        if user_locations and job_location not in user_locations:
            continue

        job_max_pay = job.get("maxPay", 0)
        if job_max_pay < min_pay:
            continue

        job_type = job.get("jobType", "").strip().lower()
        if preferred_types and job_type not in preferred_types:
            continue

        job_setting = job.get("jobSetting", "").strip().lower()
        if preferred_settings and job_setting not in preferred_settings:
            continue

        filtered.append(job)

    return filtered
