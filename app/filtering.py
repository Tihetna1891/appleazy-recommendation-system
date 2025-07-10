from typing import List

def filter_jobs(user: dict, jobs: List[dict]) -> List[dict]:
    """
    Apply filters to the job list based on the user's preferences.
    Filters are NOT applied yet — this is the ready-to-use structure.

    Expected job schema:
    - location, jobType, jobSetting, minPay, maxPay, status
    """

    # Future Example Filters (currently commented out):
    # 1. Location Match
    # user_locations = [loc.strip() for loc in user.get("location", "").split(",")]
    # jobs = [job for job in jobs if job["location"] in user_locations]

    # 2. Pay Range Overlap
    # min_pay = int(user.get("minDesiredPay", 0))
    # jobs = [job for job in jobs if job["maxPay"] >= min_pay]

    # 3. Job Type / Setting
    # preferred_type = user.get("jobType")
    # preferred_setting = user.get("jobSetting")
    # jobs = [
    #     job for job in jobs
    #     if (not preferred_type or job["jobType"] == preferred_type) and
    #        (not preferred_setting or job["jobSetting"] == preferred_setting)
    # ]

    # 4. Only Open Jobs
    # jobs = [job for job in jobs if job["status"].lower() == "open"]

    return jobs  # currently no filters applied
