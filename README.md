# ブログ

## 使用技術

- [Astro](https://astro.build)
- [Fuwari](https://github.com/saicaca/fuwari)

## 使用方法

1. 本レポジトリをクローン
2. `pnpm install` と `pnpm add sharp` を実行して依存関係をインストール
3. `pnpm new-post <filename>`で新しい記事を作成し、`src/content/posts/`.フォルダ内で編集
4. GitHub 上の `main` ブランチを更新すると、自動的にデプロイされる

## 記事のフロントマター

```yaml
---
title: My First Blog Post
published: 2023-09-09
description: This is the first post of my new Astro blog.
image: /images/cover.jpg
tags: [Foo, Bar]
category: Front-end
draft: false
---
```

## コマンド

すべてのコマンドは、ターミナルでプロジェクトのルートから実行する必要があります:

| Command                             | Action                                           |
|:------------------------------------|:-------------------------------------------------|
| `pnpm install` AND `pnpm add sharp` | 依存関係のインストール                           |
| `pnpm dev`                          | `localhost:4321`で開発用ローカルサーバーを起動      |
| `pnpm build`                        | `./dist/`にビルド内容を出力          |
| `pnpm preview`                      | デプロイ前の内容をローカルでプレビュー     |
| `pnpm new-post <filename>`          | 新しい投稿を作成                                |
| `pnpm astro ...`                    | `astro add`, `astro check`の様なコマンドを実行する際に使用 |
| `pnpm astro --help`                 | Astro CLIのヘルプを表示                     |
