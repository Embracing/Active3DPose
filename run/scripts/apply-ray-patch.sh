#!/usr/bin/env bash

SITE_PACKAGES_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" # path to /path/to/conda/envs/XXX/lib/pythonX.Y/site-packages

patch -p1 -d "${SITE_PACKAGES_DIR}/ray" << 'EOF'
--- a/rllib/evaluation/collectors/simple_list_collector.py
+++ b/rllib/evaluation/collectors/simple_list_collector.py
@@ -734,6 +734,7 @@ class SimpleListCollector(SampleCollector):
                 if data_col
                 in [
                     SampleBatch.OBS,
+                    SampleBatch.INFOS,
                     SampleBatch.ENV_ID,
                     SampleBatch.EPS_ID,
                     SampleBatch.AGENT_INDEX,
EOF
