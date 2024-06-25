---
title: Radeon RX 7900 XTでJAXを動かすまで
published: 2024-07-13
description: ''
image: ''
tags: ['JAX', 'ROCm']
category: 'Linux'
draft: false
---
## 概要

- 𝑴𝒚 𝒏𝒆𝒘 𝒈𝒆𝒂𝒓...
- Pop!\_OSで Radeon RX 7900 XT を使うためのセットアップをおこなった
- ROCm版JAXのインストール、及び動作確認をおこなった

## はじめに

![Radeon RX 7900 XT](./rx7900xt.jpg)
𝑴𝒚 𝒏𝒆𝒘 𝒈𝒆𝒂𝒓...

というわけで、Radeon RX 7900 XT を手に入れました。SAPPHIREのそこまで高くない（12万円くらい）GPUですが、VRAMが20GBもあって今後色々と活躍してくれそうです。GPUを使うにあたって、OSも心機一転、少し気になっていたPop!\_OSをクリーンインストールしました。

この記事では、Radeon用のGPUドライバやROCmのインストールから、JAXのセットアップまでの手順をまとめていきます。少しだけPop!\_OS特有の設定もありますが、DebianやUbuntu系のOSであれば、どのOSでもそこまで手順は変わらないと思うので、誰かの参考になれば嬉しいです。

## 環境

本記事は、以下の環境で検証しています。
- OS : Pop!\_OS 22.04 LTS
- CPU: Core i9-13900HX
- RAM: 32GB
- GPU: Radeon RX 7900 XT

## GPUドライバとROCmのインストール

まず、`amdgpu-install`コマンドを使えるようにします。[こちらのページ](https://www.amd.com/ja/support/linux-drivers)にアクセスし、AMD Radeonグラフィックス用のLinuxドライバーにあるUbuntu x86 64ビットから自分の環境にあったファイルをダウンロードします。（自分の場合は、Ubuntu 22.04.4用のファイルをダウンロードしました。）

ファイルのダウンロードが終わったら、ファイルを保存したディレクトリでターミナルを開き、以下のコマンドを叩きます。
```bash
sudo apt update
sudo apt install ./amdgpu-install_6.1.60103-1_all.deb
sudo apt update
```

本来は、ここで`amdgpu-install --usecase=rocm,graphics`とすればGPUドライバやROCmをインストールできるのですが、Pop!\_OSの場合、サポートしていないと怒られてしまいます。

そこで、まず以下のコマンドで`amdgpu-install`で使用されるスクリプトの場所を調べます。

```bash
which amdgpu-install
```

自分の環境だと、`/usr/bin/amdgpu-install`が出力されました。
このファイルを好きなエディタで編集します。（sudoが必要。）
```bash
sudo nano /usr/bin/amdgpu-install
```

編集箇所は、435行目以降にある`os_release`です。
```bash
os_release() {
        if [[ -r  /etc/os-release ]]; then
                . /etc/os-release
                PKGUPDATE=

                case "$ID" in
                ubuntu|linuxmint|debian|pop)
                        PKGUPDATE="apt-get update"
                        PKGMAN=apt-get
                        OS_CLASS=debian
                        :
                       ;;
                fedora|rhel|centos|almalinux|rocky|ol)
```

この関数の中にある`ubuntu|linuxmint|debian)`という行を、以下のように`ubuntu|linuxmint|debian|pop)`とします。
```bash
os_release() {
        if [[ -r  /etc/os-release ]]; then
                . /etc/os-release
                PKGUPDATE=

                case "$ID" in
                ubuntu|linuxmint|debian|pop)
                        PKGUPDATE="apt-get update"
                        PKGMAN=apt-get
                        OS_CLASS=debian
                        :
                       ;;
                fedora|rhel|centos|almalinux|rocky|ol)
```

編集し終わったら、変更を保存します。その後、ターミナルで以下のコマンドを叩くとGPUドライバとROCmのインストールが始まります。（`--no-dkms`は必要らしいという情報は見たのですが、本当に`--no-dkms`がないとインストールが失敗するのかは未確認です。）
```bash
amdgpu-install --usecase=rocm,graphics --no-dkms
```

GPUドライバとROCmのインストールが終わったら、以下のコマンドを叩きます。
```bash
sudo usermod -a -G render,video $LOGNAME
```

ここで、PCを一度再起動しましょう。再起動完了後、`rocm-smi`コマンドを叩いてみてGPUの情報が出力されれば、インストール成功です。

## JAXのインストール

JAXを動かすには、jaxlibとjaxの2つのライブラリが必要です。インストールの手順としては、まずjaxlibを入れて、その後jaxを入れるという流れになります。

ROCm版のjaxlibは、[こちらのページ](https://github.com/ROCm/jax/releases)でビルド済みのものが配布されています。これをインストールすれば動いてくれるはずなのですが、私の環境だとうまく動いてくれませんでした。そこで、jaxlibに関しては[ROCm/jax](https://github.com/ROCm/jax)で管理されているソースコードを自分でビルドすることにしました。

jaxlibのビルドを始める前に、`libstdc++-12-dev`をインストールします。（これがないと、"cmathが無いよ！"とビルド中に怒られます。）
```bash
sudo apt-get install libstdc++-12-dev
```

次に、[ROCm/jax](https://github.com/ROCm/jax)をクローンして、インストールしたいバージョンのタグに切り替えます。（今回は、v0.4.30をインストールしていきます。）
```bash
git clone https://github.com/ROCm/jax.git
cd jax
git checkout -b rocm-jaxlib-v0.4.30 refs/tags/rocm-jaxlib-v0.4.30
```

その後、jaxlibのビルドをおこないます。ただし、rocm_path, python_versionはそれぞれの環境に合わせて適宜変更してください。注意点として、自分の環境だとビルドに最大で32GB程のRAMを使っていました。もしメモリにあまり余裕がないPCを使っている場合は、ブラウザなどは起動しないほうがいいかもしれないです。
```bash
python build/build.py --enable_rocm --rocm_path=/opt/rocm-6.1.3 --python_version=3.12
```

無事にビルドが終了すると、`dist/`配下にwhlファイルが保存されるので、これをpipやPoetryなどを使ってインストールします。例えば、pipでインストールする場合は以下のようになります。
```bash
pip install dist/jaxlib-0.4.30.dev20240713-cp312-cp312-manylinux2014_x86_64.whl
```

最後に、jaxをインストールします。jaxに関しては、配布されているものを直接インストールしても問題なく動きました。注意点としては、jaxlibとjaxのバージョンをちゃんと合わせることくらいでしょうか。
```bash
pip install jax==0.4.30
```

## 動作確認

jaxとjaxlibのインストールが完了したら簡単に動作確認をしましょう。
Pythonインタプリタ内で以下のようにコマンドを叩いてみて、同様の結果が返ってくればOKです。

```bash
>>> import jax
>>> jax.devices()
[rocm(id=0)]
>>> jax.numpy.zeros(1)
Array([0.], dtype=float32)
>>> jax.random.normal(jax.random.PRNGKey(0))
Array(-0.20584226, dtype=float32)
```

## 追記（2024.7.15）

[ROCm/jax](https://github.com/ROCm/jax)のクローンとjaxlibのビルドをおこなうスクリプトを作成し、[h-terao/rocm-jax-wheels](https://github.com/h-terao/rocm-jax-wheels)に公開しました。また、個人用にビルドしたwheelもリリースページで公開しているので、ROCmやjaxlibのバージョンが欲しい物と一致している場合は試してみてください。
