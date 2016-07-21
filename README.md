![What I Wish I Knew When Learning Haskell](https://raw.githubusercontent.com/sdiehl/wiwinwlh/master/img/title.png)

# 私がHaskellの勉強中に知りたかったこと

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

* [反駁不可パターン](遅延性#irrefutable-patterns)
* [Hackage](基本#hackage)
* [網羅性](基本#exhaustiveness)
* [スタックトレース](基本#stacktraces)
* [遅延性](遅延性)
* [型システム](量化#type-systems)（スコーレム・キャプチャ）
* （他言語）[関数ポインタ](FFI#function-pointers)
* [attoparsec](パース処理#attoparsec)
* （インラインの）[Cmm](GHC#cmm)
* [IO と ST](GHC#io-st)（PrimMonad）
* [特殊化](GHC#specialization)
* [ジェネリクス版 unbound](言語#unbound-generics)
* [エディタの統合](基本#editor-integration)
* [EKG](プロファイリング#ekg)
* [Nix](基本#nix)
* [Haddock](基本#haddock)
* [モナド・チュートリアル](モナド#monad-tutorials)
* [モナド射](モナド変換子#monad-morphisms)
* [余再帰](Prelude#corecursion)
* [圏](Applicative#category)
* [アロー](Applicative#arrows)
* [双関手](Applicative#bifunctors)
* [ExceptT](エラー処理#exceptt)
* [hint と mueval](インタプリタ#hint-and-mueval)
* [役割](型族#roles)
* [高階種](昇格#higher-kinds)
* [種多相](昇格#kind-polymorphism)
* [数値の塔](数学#numeric-tower)
* [SAT ソルバ](数学#sat-solvers)
* [グラフ](データ構造#graphs)
* [スパーク](並行性#sparks)
* [スレッドスコープ](並行性#threadscope)
* [ジェネリックなパース処理](パース処理#generic-parsing)
* [ブロック図](GHC#block-diagram)（とデバッグ用フラグ）
* [コア](GHC#core)
* [インライナ](GHC#inliner)
* [非ボックス型](GHC#unboxed-types)（と実行時メモリ表現）
* [ghc-heap-view](GHC#ghc-heap-view)
* [STG](GHC#stg)
* [ワーカとラッパ](GHC#worker-wrapper)
* [Z エンコーディング](GHC#z-encoding)
* [Cmm](GHC#cmm)
* [最適化ハック](GHC#optimization-hacks)
* [RTS プロファイリング](プロファイリング#rts-profiling)
* [代数的関係](圏#algebraic-relations)
