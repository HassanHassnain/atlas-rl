"""Built-in DevOps/SRE environments. Importing this package registers all envs."""

from atlas_rl.envs import (  # noqa: F401
    log_triage,
    config_repair,
    ci_doctor,
    runbook_planner,
    shell_golf,
    cron_author,
    regex_extract,
    dockerfile_lint,
    k8s_doctor,
    semver_resolve,
)
