---
title: apt updateでBraveブラウザに関するエラーが出たときの対処法
published: 2024-07-14
description: ''
image: ''
tags: []
category: 'Linux'
draft: false
---

## 問題

Pop!\_OSにBraveブラウザをインストールしました。
すると、`apt update`で、以下のエラーが出るようになりました。

> N: リポジトリ 'https://brave-browser-apt-release.s3.brave.com stable InRelease' がアーキテクチャ 'i386' をサポートしないため設定ファイル 'main/binary-i386/Packages' の取得をスキップ

他のパッケージは問題なく動いているので実害はほとんど無さそうです。ただ、なんとなく気持ち悪いのでこのエラーが出ないようにしました。

## 対処法

調べていると、[このページ](https://askubuntu.com/questions/1453435/apt-update-gives-error-on-brave-browser)でまったく同じ質問されていました。対処法としては、`/etc/apt/sources.list.d/brave-browser-release.list`に`arch=amd64`というパラメータを追加すれば良いそうです。

以下のコマンドを叩けば、既に存在する`brave-browser-release.list`を`arch=amd64`付きのものに入れ替えてくれます。
```bash
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main"|sudo tee /etc/apt/sources.list.d/brave-browser-release.list
```

`brave-browser-release.list`を更新したあと、再度`apt update`したらエラーが出力されないようになりました。

## 参考

- [askubuntu: apt update gives error on brave browser](https://askubuntu.com/questions/1453435/apt-update-gives-error-on-brave-browser)
