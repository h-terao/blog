---
title: "JSONの辞書は重複したキーを禁止していない"
published: 2024-06-20
description: ""
tags: []
category: 小ネタ
draft: false
---

## はじめに

JSONは、データ構造を文字列で表現するためのフォーマットです。
文字列はほとんどのプログラミング言語で容易に取り扱うことができるため、データをプログラミング間でやり取りする際、非常に便利です。
一方、JSONはただの文字列であるが故に、本来は許容されないはずのデータ構造を表すことも理論上はできます。

ある日、JSONで辞書データを表現する際、重複したキーを表現できてしまうのでは？という点が気になりました。本記事では、この点について調査し、分かったことをまとめていきます。

## JSONの仕様を確認する

JSONの仕様は[RFC8259](https://datatracker.ietf.org/doc/html/rfc8259)に書かれています。
この文書の4章には、以下のような記述があります。

> An object structure is represented as a pair of curly brackets
> surrounding zero or more name/value pairs (or members).  A name is a
> string.  A single colon comes after each name, separating the name
> from the value.  A single comma separates a value from a following
> name.  The names within an object SHOULD be unique.

ここで問題になるのは、最後の文

> The names within an object SHOULD be unique.

にある"SHOULD"が強制なのか推奨なのか、どちらの意味で使われているかです。
[RFC2119](https://www.ietf.org/rfc/rfc2119.txt?number=2119)によると、SHOULDは以下のように定義されています。

> SHOULD   This word, or the adjective "RECOMMENDED", mean that there
   may exist valid reasons in particular circumstances to ignore a
   particular item, but the full implications must be understood and
   carefully weighed before choosing a different course.

この定義を読むと分かる通り、SHOULDは推奨の意味で用いられています。
つまり、JSONで辞書を表す際、キーが重複するのは推奨されていませんが、明確に禁止もされていません。

## プログラミング言語での取り扱い

JSONの辞書がキーの重複を許していたとしても、ほとんどのプログラミング言語では辞書に同じキーを用いることはできません。
それでは、重複したキーを持つ辞書をJSONにしてプログラミング言語に渡すとどういった挙動を示すのでしょうか？
以下、Pythonで試してみた結果をまとめます。
（本当はもっと増やしたい😓）

### Python

以下のコードをインタプリタで叩いてみると、

```python
import json
json.loads('{"x": 1, "x": 2}')
```

{'x': 2}という辞書が返却されました。
Pythonの場合、後ろの値が優先的に用いられるようです。

## 参考
- [関連する質問（Stack Overflow）](https://stackoverflow.com/questions/5306741/do-json-keys-need-to-be-unique)
- [RFC8259](https://datatracker.ietf.org/doc/html/rfc8259)
- [RFC2119](https://www.ietf.org/rfc/rfc2119.txt?number=2119)