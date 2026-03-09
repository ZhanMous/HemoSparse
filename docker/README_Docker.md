# 使用 Docker 在容器中生成 PPTX → PDF

构建镜像（在仓库根目录运行）：

```bash
docker build -t hemosparse-pptx -f HemoSparse/Dockerfile .
```

运行容器（会在容器内执行转换并将结果放在 `HemoSparse/outputs/`）：

```bash
docker run --rm -v "$(pwd)/HemoSparse:/workspace/HemoSparse" hemosparse-pptx
```

转换完成后，PDF 位于 `HemoSparse/outputs/HemoSparse_presentation.pdf`。

注意：镜像较大（含 LibreOffice），下载与构建需要一定时间。
