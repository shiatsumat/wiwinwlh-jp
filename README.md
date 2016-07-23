![What I Wish I Knew When Learning Haskell](https://raw.githubusercontent.com/sdiehl/wiwinwlh/master/img/title.png)

# 私がHaskellの勉強中に知りたかったこと

バージョン: 2.2

## 執筆協力

この文献には原文からの翻訳ミスや誤字・分かりづらい表現などが存在するかもしれません。その場合、ぜひ修正して[リクエスト](https://github.com/Kinokkory/wiwinwlh-jp/pulls)を送ってください！

この文献は[GitBook](https://www.gitbook.com/)によって管理されています。

## このプロジェクトをローカルで動かすには

[node](https://nodejs.org/en/)をインストールしてください。それから、[GitBook](https://www.gitbook.com/)のコマンドラインツールを以下の手順でインストールします。

```bash
npm -g install gitbook-cli
```

それから、以下の手順でローカルにサーバを立てます。

```bash
gitbook install
gitbook serve
```

ブラウザでコマンド実行時に出てきたURLを、表示してください。

## クレジット

このページはスティーヴン・ディール([@smdiehl](https://twitter.com/smdiehl))による[What I Wish I Knew When Learning Haskell](http://dev.stephendiehl.com/hask/)の日本語訳です。

現在、第3草稿のものが翻訳されています。原文は現在第4草案です。最新の文献が確認したい場合、原文を参照してください。

原文の全てのコードのソースは[ここ](https://github.com/sdiehl/wiwinwlh/tree/master/src)にあります。誤りやより分かりやすい例があれば、[GitHub](https://github.com/sdiehl/wiwinwlh)にプルリクエストを気兼ねなく出してください。

## ライセンス

著作権はありません。このプロジェクトに関わる人全てがこの作品に関する権利の全てを放棄することにより、このコードとテキストはパブリックドメインに捧げられています。

たとえ商業目的であったとしても、この作品の複製・変更・頒布・実演を許可なく行うことができます。

## 更新履歴

### 2.2

追記・大きい変更のなされた節：

* [反駁不可パターン](docs/laziness.md#反駁不可パターン)
* [Hackage](docs/basics.md#hackage)
* [網羅性](docs/basics.md#網羅性)
* [スタックトレース](docs/basics.md#スタックトレース)
* [遅延性](docs/laziness.md)
* [型システム](docs/quantification.md#型システム)（スコーレム・キャプチャ）
* （他言語）[関数ポインタ](docs/ffi.md#関数ポインタ)
* [attoparsec](docs/parsing.md#attoparsec)
* （インラインの）[Cmm](docs/ghc.md#cmm)
* [IOとST](docs/ghc.md#ioとst)（PrimMonad）
* [特殊化](docs/ghc.md#特殊化)
* [ジェネリクス版unbound](docs/languages.md#ジェネリクス版unbound)
* [エディタの統合](docs/basics.md#エディタの統合)
* [EKG](docs/profiling.md#ekg)
* [Nix](docs/basics.md#nix)
* [Haddock](docs/basics.md#haddock)
* [モナド・チュートリアル](docs/monads.md#モナド・チュートリアル)
* [モナド射](docs/monad-transformers.md#モナド射)
* [余再帰](docs/prelude.md#余再帰)
* [圏](docs/applicatives.md#圏)
* [アロー](docs/applicatives.md#アロー)
* [双関手](docs/applicatives.md#双関手)
* [ExceptT](docs/error-handling.md#exceptt)
* [hintとmueval](docs/interpreters.md#hintとmueval)
* [役割](docs/type-families.md#役割)
* [高階種](docs/promotion.md#高階種)
* [種多相](docs/promotion.md#種多相)
* [数値の塔](docs/mathematics.md#数値の塔)
* [SATソルバ](docs/mathematics.md#satソルバ)
* [グラフ](docs/unordered-containers.md#グラフ)
* [スパーク](docs/concurrency.md#スパーク)
* [スレッドスコープ](docs/concurrency.md#スレッドスコープ)
* [ジェネリックなパース処理](docs/parsing.md#ジェネリックなパース処理)
* [ブロック図](docs/ghc.md#ブロック図)（とデバッグ用フラグ）
* [コア](docs/ghc.md#コア)
* [インライナ](docs/ghc.md#インライナ)
* [非ボックス型](docs/ghc.md#非ボックス型)（と実行時メモリ表現）
* [ghc-heap-view](docs/ghc.md#ghc-heap-view)
* [STG](docs/ghc.md#stg)
* [ワーカとラッパ](docs/ghc.md#ワーカとラッパ)
* [Zエンコーディング](docs/ghc.md#zエンコーディング)
* [Cmm](docs/ghc.md#cmm)
* [最適化ハック](docs/ghc.md#最適化ハック)
* [RTSプロファイリング](ocs/profiling.md#rtsプロファイリング)
* [代数的関係](docs/categories.md#代数的関係)
