# The engine image (both ranks): TensorFold v1.1.2's native (Zig) server, tensorfold-native, built for the GB10 (sm_121) from
# the upstream checkout at TF_REF (the commit the v1.1.2 tag points to), plus patches/*.patch if there are any (git apply in
# the checkout's root, filename order). Upstream v1.1.2 has NO minimax_m3_vl family: an image built with patches/ empty
# serves other models but not this checkpoint, and start.sh / prepare.sh refuse to go on with it (family gate).
# It builds on NVIDIA's PyTorch container, as the sibling DeepSeek-V4.1 recipe does: its CUDA toolchain (nvcc) compiles the
# sm_121 fatbins that the server embeds, and its NCCL and libibverbs are what the engine loads at run time. No GPU is needed
# to build. TensorFold's CUDA build has no other dependency (build.zig.zon lists none). The build line is the one that built
# tensorfold-native 1.1.2 for sm_121 on an aarch64 Linux host with nvcc 13.0.88 (without --prefix) (-Doptimize=ReleaseSafe); inside this
# image, with this base, it has not been run yet.
# Built by scripts/prepare.sh, never by hand: it passes the build arguments and a build context of this file and patches/ only.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG SM=121

# The image decoders the engine loads at run time (mm3/vision; TensorFold ships neither): JPEG through libjpeg-turbo's
# TurboJPEG library and WebP through libwebp's decoder library, at the releases the engine pins and checks by version
# (zig/src/core/picture: jpeg.zig "3.1.4.1", webp.zig 0x010600), built here from the release tarballs (sha256 below) and
# named by TENSORFOLD_LIBTURBOJPEG / TENSORFOLD_LIBWEBP. PNG needs no library. Their licenses go in /opt/imagelibs/share.
ARG LIBJPEG_TURBO_VERSION=3.1.4.1
ARG LIBJPEG_TURBO_SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
ARG LIBWEBP_VERSION=1.6.0
ARG LIBWEBP_SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
RUN mkdir -p /tmp/imagelibs /opt/imagelibs/lib /opt/imagelibs/share/licenses && cd /tmp/imagelibs && \
    curl -fsSL -o jpeg.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz" && \
    echo "${LIBJPEG_TURBO_SHA256}  jpeg.tar.gz" | sha256sum -c - && tar -xzf jpeg.tar.gz && \
    cmake -S "libjpeg-turbo-${LIBJPEG_TURBO_VERSION}" -B jpeg-build -DCMAKE_BUILD_TYPE=Release -DENABLE_STATIC=0 \
      -DWITH_TURBOJPEG=1 -DWITH_TOOLS=0 -DCMAKE_INSTALL_PREFIX=/tmp/imagelibs/jpeg -DCMAKE_INSTALL_LIBDIR=lib && \
    cmake --build jpeg-build -j"$(nproc)" && cmake --install jpeg-build && \
    cp -a /tmp/imagelibs/jpeg/lib/libturbojpeg.so.0* /opt/imagelibs/lib/ && \
    grep -aq "libjpeg-turbo version ${LIBJPEG_TURBO_VERSION}" /opt/imagelibs/lib/libturbojpeg.so.0 && \
    cp "libjpeg-turbo-${LIBJPEG_TURBO_VERSION}/LICENSE.md" /opt/imagelibs/share/licenses/libjpeg-turbo-LICENSE.md && \
    cp "libjpeg-turbo-${LIBJPEG_TURBO_VERSION}/README.ijg" /opt/imagelibs/share/licenses/libjpeg-turbo-README.ijg && \
    curl -fsSL -o webp.tar.gz "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-${LIBWEBP_VERSION}.tar.gz" && \
    echo "${LIBWEBP_SHA256}  webp.tar.gz" | sha256sum -c - && tar -xzf webp.tar.gz && \
    cd "libwebp-${LIBWEBP_VERSION}" && \
    ./configure --prefix=/tmp/imagelibs/webp --disable-static --enable-libwebpdecoder --disable-libwebpmux \
      --disable-libwebpdemux --disable-png --disable-jpeg --disable-tiff --disable-gif --disable-gl --disable-sdl --disable-wic && \
    make -j"$(nproc)" && make install && cd /tmp/imagelibs && \
    cp -a /tmp/imagelibs/webp/lib/libwebpdecoder.so.3* /opt/imagelibs/lib/ && \
    python3 -c "import ctypes; v = ctypes.CDLL('/opt/imagelibs/lib/libwebpdecoder.so.3').WebPGetDecoderVersion(); assert v == 0x010600, hex(v)" && \
    cp "libwebp-${LIBWEBP_VERSION}/COPYING" /opt/imagelibs/share/licenses/libwebp-COPYING && \
    cp "libwebp-${LIBWEBP_VERSION}/PATENTS" /opt/imagelibs/share/licenses/libwebp-PATENTS && \
    cd / && rm -rf /tmp/imagelibs
ENV TENSORFOLD_LIBTURBOJPEG=/opt/imagelibs/lib/libturbojpeg.so.0 TENSORFOLD_LIBWEBP=/opt/imagelibs/lib/libwebpdecoder.so.3

# the serving engine
ARG TF_REPO
ARG TF_REF
ARG TF_TAG
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}" && \
    test "$(cat .zig-version)" = "${ZIG_VERSION}" && \
    grep -q "\.version = \"${TF_TAG#v}\"" build.zig.zon
COPY patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# libtfgrammar (structured output, tool_choice "required" / named) is an optional runtime library built from an xgrammar
# checkout at the tag the engine pins (zig/build/grammar.zig); without it the server refuses those requests with 400. The
# server looks for it in lib beside its bin (native/lib), so it moves there from the prefix's lib.
ARG XGRAMMAR_TAG=v0.2.8
RUN cd /tensorfold && /opt/zig/zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Dversion="${TF_TAG#v}" -Doptimize=ReleaseSafe \
      --prefix /opt/tensorfold -j"$(nproc)" && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && \
    grep -q "pinned = \"${XGRAMMAR_TAG}\"" zig/build/grammar.zig && \
    git clone -q --depth 1 --branch "${XGRAMMAR_TAG}" --recurse-submodules --shallow-submodules \
      https://github.com/mlc-ai/xgrammar /opt/xgrammar && \
    /opt/zig/zig build grammar -Dxgrammar=/opt/xgrammar --prefix /opt/tensorfold -j"$(nproc)" && \
    mkdir -p /opt/tensorfold/native/lib && mv /opt/tensorfold/lib/libtfgrammar.so /opt/tensorfold/native/lib/ && \
    { rmdir /opt/tensorfold/lib 2>/dev/null || true; } && \
    test -f /opt/tensorfold/native/lib/libtfgrammar.so && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold/LICENSES && \
    cp /opt/xgrammar/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-Apache-2.0.txt && \
    cp /opt/xgrammar/NOTICE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-NOTICE.txt && \
    cp /opt/xgrammar/3rdparty/dlpack/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/dlpack-Apache-2.0.txt && \
    for f in LICENSE NOTICE THIRD_PARTY_NOTICES.md LICENSES; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done && \
    rm -rf /tensorfold /opt/xgrammar /root/.cache/zig

ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.engine=zig tf.ref=${TF_REF} tf.tag=${TF_TAG}
ENV PATH=/opt/tensorfold/native/bin:${PATH}
WORKDIR /workspace
