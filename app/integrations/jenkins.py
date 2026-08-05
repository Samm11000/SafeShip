"""
Jenkins integration instructions.

Jenkins is the richest of the three — it is the only platform that exposes test
results through its own API — and the fiddliest, because nothing is granted
automatically. It needs python3 on the agent and an API token to read history.

This replaces the Groovy that used to be built inline in dashboard.py, which
hardcoded seven of the ten features.
"""
from __future__ import annotations

ID = "jenkins"
NAME = "Jenkins"
TAGLINE = "Works with firewalled and on-prem controllers. Reads your test results natively."
DOCS_URL = "https://www.jenkins.io/doc/book/using/using-credentials/"
SETUP_EFFORT = "~5 minutes"

CLONE_URL = "https://github.com/Samm11000/SafeShip"


def describe(tenant_id, api_key, base_url):
    return {
        "id": ID,
        "name": NAME,
        "tagline": TAGLINE,
        "docs_url": DOCS_URL,
        "setup_effort": SETUP_EFFORT,
        "secrets_location": "Manage Jenkins → Credentials → System → Global credentials → Add Credentials → Secret text",
        "secrets": [
            {"name": "SAFESHIP_TENANT_ID", "value": tenant_id,
             "where": "Manage Jenkins → Credentials (Secret text)"},
            {"name": "SAFESHIP_API_KEY", "value": api_key,
             "where": "Manage Jenkins → Credentials (Secret text)"},
            {"name": "SAFESHIP_URL", "value": base_url,
             "where": "Manage Jenkins → Credentials, or the job environment"},
        ],
        "prerequisites": [
            {
                "title": "Add JENKINS_USER and JENKINS_TOKEN",
                "why": "Jenkins hands a build no credentials of its own, so reading "
                       "your job's history needs an API token. Without it the API "
                       "returns 403 and recent_failure_rate, days_since_deploy and "
                       "build_time_delta are all estimated. Use an API token, never "
                       "a password.",
                "fix": "Your user → Configure → API Token → Add new token,\n"
                       "then expose it to the job as JENKINS_USER + JENKINS_TOKEN.",
            },
            {
                "title": "python3 on the agent",
                "why": "safeship_ci is stdlib-only, so python3 is the whole "
                       "requirement — there is nothing to pip install and no "
                       "dependency that can conflict with your build's.",
                "fix": "python3 --version        # 3.9 or newer",
            },
            {
                "title": "Do not shallow-clone",
                "why": "If the Git SCM step uses a shallow clone, HEAD~1 is absent "
                       "and diff_size and files_changed cannot be measured.",
                "fix": "In the job's Git SCM config, remove 'Shallow clone' "
                       "or set depth to at least 2.",
            },
            {
                "title": "Publish your test results",
                "why": "Jenkins is the one platform with a native test-results API. "
                       "Publishing with the junit step lets SafeShip read "
                       "test_pass_rate directly from Jenkins.",
                "fix": "junit 'reports/junit.xml'",
            },
        ],
        "history_note": "Build history is read from your Jenkins over its own API, "
                        "using your token, from inside your network. SafeShip needs "
                        "no inbound access to your controller.",
        "snippet": {
            "filename": "Jenkinsfile",
            "language": "groovy",
            "code": _jenkinsfile(),
        },
        "verify_hint": "Run the job once. The risk gate stage prints the score and "
                       "this page flips to Connected.",
    }


def _jenkinsfile():
    # Literal replace rather than %-formatting: see the note in github.py.
    return _JENKINSFILE.replace("__CLONE__", CLONE_URL)


_JENKINSFILE = """\
// Add these stages to your existing Jenkinsfile.
pipeline {
  agent any

  environment {
    SAFESHIP_URL       = credentials('SAFESHIP_URL')
    SAFESHIP_TENANT_ID = credentials('SAFESHIP_TENANT_ID')
    SAFESHIP_API_KEY   = credentials('SAFESHIP_API_KEY')
    SAFESHIP_PKG       = '/tmp/safeship-ci'

    // Needed for build history. Without these, three features are estimated.
    JENKINS_USER  = credentials('JENKINS_USER')
    JENKINS_TOKEN = credentials('JENKINS_TOKEN')
  }

  stages {
    stage('Fetch safeship_ci') {
      steps {
        sh '''
          if [ ! -d "$SAFESHIP_PKG/safeship_ci" ]; then
            git clone --depth 1 __CLONE__ "$SAFESHIP_PKG"
          else
            git -C "$SAFESHIP_PKG" pull --ff-only || true
          fi
        '''
      }
    }

    stage('SafeShip risk gate') {
      steps {
        // Exits non-zero ONLY on a BLOCKED verdict. An outage, timeout or 500
        // warns and exits 0 — the gate never blocks a deploy because the gate
        // itself is broken. Drop --fail-open to start enforcing.
        sh '''
          PYTHONPATH="$SAFESHIP_PKG" python3 -m safeship_ci score --fail-open
        '''
      }
    }
  }

  post {
    // The learning signal. Without it the model never sees a real outcome.
    success { sh 'PYTHONPATH="$SAFESHIP_PKG" python3 -m safeship_ci log 0 || true' }
    failure { sh 'PYTHONPATH="$SAFESHIP_PKG" python3 -m safeship_ci log 1 || true' }
  }
}
"""
