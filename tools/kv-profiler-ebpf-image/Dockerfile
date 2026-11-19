FROM sha256:c25c4f62ca7bc4433e730fc8a6238532dbe8739f13e62018a503f2c43c200dd5 AS build
COPY . /freeze
RUN /bin/bash /freeze/build-tools.sh

FROM sha256:c25c4f62ca7bc4433e730fc8a6238532dbe8739f13e62018a503f2c43c200dd5
COPY --from=build /opt/kv-profiler-tools /opt/kv-profiler-tools
LABEL local.kv-profiler.tooling="bpftool-v7.6.0-clang-18-frozen"
