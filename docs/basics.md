# 基本

## Cabal

Cabal は Haskell のビルドシステムであり、パッケージマネージャの役割も兼ね備えています。

例えば、Hackage から自分のシステムに [parsec](http://hackage.haskell.org/package/parsec) をインストールするには、install コマンドを呼び出してください。

```bash
$ cabal install parsec           # 最新バージョン
$ cabal install parsec==3.1.5    # バージョンの指定
```

Haskell のパッケージの通常のビルドの呼び出しは以下の通り。

```bash
$ cabal get parsec    # 取ってくるソース
$ cd parsec-3.1.5

$ cabal configure
$ cabal build
$ cabal install
```

Hackage からパッケージのインデクスを更新するには以下を実行。

```bash
$ cabal update
```

新しい Haskell のプロジェクトを始めるには以下を実行。

```bash
$ cabal init
$ cabal configure
```

``.cabal`` ファイルが新しいプロジェクトのための設定とともに作られます。

Cabal の最新機能はサンドボックスの追加です（1.18 以降の cabal）。これは、Haskell パッケージの自己完結型の環境です。グローバルのパッケージ・インデクスとは別に、プロジェクトのルートの ``./.cabal-sandbox`` にあります。新しいサンドボックスを cabal プロジェクトに作るには、以下を実行。

```bash
$ cabal sandbox init
```

また、サンドボックスは潰すこともできます。

```bash
$ cabal sandbox delete
```

サンドボックスの設定があるプロジェクトの運用しているディレクトリで cabal のコマンドを呼び出すと、cabal 自体の振る舞いが変化します。例えば、``cabal install`` コマンドはローカルなパッケージ・インデクスのみを変え、グローバルな設定には触れません。

cabal ファイルから新規に作ったサンドボックスに依存関係を組み込むには以下を実行。

```bash
$ cabal install --only-dependencies
```

``-j<n>`` を渡せば、依存関係を並列的に組み立てることもできます。ただし ``n`` には並行するビルドの数を入れます。

```bash
$ cabal install -j4 --only-dependencies
```

サンプルの cabal ファイルを見てみましょう。ここには、いかなるパッケージも提供しうる 2 つのエントリポイントがあります。``library``（ライブラリ）と ``executable``（実行可能物）です。複数の ``executable`` を定義することは可能ですが、``library`` は一つしか定義できません。特殊な形の実行可能なエントリポイントとして ``Test-Suite`` もあります。これは、cabal から呼び出されるユニットテストのインターフェースを定義するものです。

ライブラリに対しては、cabal ファイルの ``exposed-modules`` フィールドが、パッケージの構造の中のどのモジュールがパッケージのインストール時に表に見えるかを示しています。これらのモジュールは、下流の消費者に公開したいと私たちが思っている、ユーザーが目にする API だからです。

実行可能物に対しては、``main-is`` フィールドが Main モジュールを示しています。``main`` 関数をエクスポートしているプロジェクトがアプリケーションの実行可能な論理を見つけられるようにするためです。

```bash
name:               mylibrary
version:            0.1
cabal-version:      >= 1.10
author:             Paul Atreides
license:            MIT
license-file:       LICENSE
synopsis:           The code must flow.
category:           Math
tested-with:        GHC
build-type:         Simple

library
    exposed-modules:
      Library.ExampleModule1
      Library.ExampleModule2

    build-depends:
      base >= 4 && < 5

    default-language: Haskell2010

    ghc-options: -O2 -Wall -fwarn-tabs

executable "example"
    build-depends:
        base >= 4 && < 5,
        mylibrary == 0.1
    default-language: Haskell2010
    main-is: Main.hs

Test-Suite test
  type: exitcode-stdio-1.0
  main-is: Test.hs
  default-language: Haskell2010
  build-depends:
      base >= 4 && < 5,
      mylibrary == 0.1
```

ライブラリの”実行可能物”を cabal サンドボックス下で実行するには：

```bash
$ cabal run
$ cabal run <name>
```

GHCi シェルに”ライブラリ”を cabal サンドボックス下で読み込むには：

```bash
$ cabal repl
$ cabal repl <name>
```

``<name>`` メタ変数は、cabal ファイルの実行可能物でもライブラリ宣言でも構いませんし、オプションでプレフィクス ``exe:<name>`` や ``lib:<name>`` をそれぞれの場合について付けて、曖昧さを無くすこともできます。

パッケージを ``./dist/build`` フォルダにローカルに作るには、``build`` コマンドを実行します。

```bash
$ cabal build
```

テストを動かすには、パッケージ自体を再び ``--enable-tests`` を付けて構成せねばならず、Test-Suite から求められている ``build-depends`` は、まだの場合、手動でインストールせねばなりません。

```bash
$ cabal install --only-dependencies --enable-tests
$ cabal configure --enable-tests
$ cabal test
$ cabal test <name>
```

また、GHC の環境変数がサンドボックスのために設定された状態で、任意のシェルコマンドを呼び出すこともできます。一般的には、``ghc`` や ``ghci`` コマンドがサンドボックスを使えるように、このコマンドとともに新しいシェルを呼び出すのだ。（デフォルトではサンドボックスが使われないので、よく不満の種となっている。）

```bash
$ cabal exec
$ cabal exec sh # GHC サンドボックスのパスを設定して、シェルを起動する。
```

``haddock`` コマンドを実行することで、Haddock の文書をローカルなプロジェクトのために作ることができます。Haddock の文書は ``./dist`` フォルダに作られます。

```bash
$ cabal haddock
```

Hackage にアップロードする準備がやっとできたならば（Hackage のアカウントは作ってあるとして）、以下のコマンドでtarball を作り、アップロードすることができます。

```bash
$ cabal sdist
$ cabal upload dist/mylibrary-0.1.tar.gz
```

時に、ライブラリをローカルなプロジェクトからサンドボックスへと追加したい、と思うこともあるでしょう。この場合、add-source コマンドを使って、ローカルなディレクトリからサンドボックスへとそのライブラリを持って行くことができます。

```bash
$ cabal sandbox add-source /path/to/project
```

サンドボックスの現在の状態は、全ての現在のパッケージの制約を列挙した状態で、凍結することができます。

```bash
$ cabal freeze
```

これにより、制約の集合の書かれた ``cabal.config`` ファイルが作られます。

```haskell
constraints: mtl ==2.2.1,
             text ==1.1.1.3,
             transformers ==0.4.1.0
```

``cabal repl`` と ``cabal run`` コマンドを使うことが望ましいですが、シェルで同様のことを手動でおこないたい場合もあります。そんな時には、いくつかの便利なエイリアスがあります。これらは、シェルのディレクトリの展開を利用して、カレントディレクトリにあるパッケージのデータベースを見つけ、GHC を適切なフラグとともに立ち上げます。

```bash
alias ghc-sandbox="ghc -no-user-package-db -package-db .cabal-sandbox/*-packages.conf.d"
alias ghci-sandbox="ghci -no-user-package-db -package-db .cabal-sandbox/*-packages.conf.d"
alias runhaskell-sandbox="runhaskell -no-user-package-db -package-db .cabal-sandbox/*-packages.conf.d"
```

シェルのカレントディレクトリのサンドボックスの状態を表示する zsh スクリプトもあります。

```bash
function cabal_sandbox_info() {
    cabal_files=(*.cabal(N))
    if [ $#cabal_files -gt 0 ]; then
        if [ -f cabal.sandbox.config ]; then
            echo "%{$fg[green]%}sandboxed%{$reset_color%}"
        else
            echo "%{$fg[red]%}not sandboxed%{$reset_color%}"
        fi
    fi
}

RPROMPT="\$(cabal_sandbox_info) $RPROMPT"
```

cabal の設定は ``$HOME/.cabal/config`` に入っていて、Hackage へのアップロードのための信用証明情報を含む、様々な選択肢があります。設定への一つの追加は、衝突が生じてしまうのを防ぐために、サンドボックス外でのパッケージのインストールを全面禁止するというものです。

```perl
-- パッケージのグローバルなインストールを禁止する。
require-sandbox: True
```

ライブラリは、実行時のプロファイリング情報を有効にしてコンパイルすることもできます。これについては、[並行性](#concurrency)と[プロファイリング](#profiling)の節でより詳しく触れています。

```perl
library-profiling: True
```

有効にするフラグとしてこれら以外で一般的なものは、``documentation`` です。これは、Haddock 文書をローカルで作るようにするもので、オフラインの参照に便利なことがあります。文書が作られる場所は、Linux のファイルシステムでは、``/usr/share/doc/ghc/html/libraries/`` ディレクトリです。

```perl
documentation: True
```

もし GHC が現在インストールされているならば、以下のローカルリンクで Prelude と Base ライブラリの文書が見られます。

[/usr/share/doc/ghc/html/libraries/index.html](file:///usr/share/doc/ghc/html/libraries/index.html)

参照：

* [An Introduction to Cabal Sandboxes](http://coldwa.st/e/blog/2013-08-20-Cabal-sandbox.html)
* [Storage and Identification of Cabalized Packages](http://www.vex.net/~trebla/haskell/sicp.xhtml)

## Hackage

Hackage はオープンソースの Haskell パッケージの標準的なソースです。Haskell は変化している言語であるため、Hackage が意味するものは人によりけりですが、Hackage では2つの哲学が支配しているように思います。

**再利用可能なコード／積み木**

ライブラリは安定した、コミュニティーに支えられた、普及し安定している体系の上に高レベルの機能を作るための積み木として存在しています。ライブラリの作者は、問題領域に対する彼らの理解をまとめ、他の人がその理解と専門知識を足場とできるようにするための手段として、ライブラリを書いています。

**準備場所／コメントの要求**

広く抱かれている哲学は、Hackage は実験的なライブラリをアップロードして、コミュニティのフィードバックを得て、コードを誰でも利用できるようにするための場所だというものです。ライブラリの作者はこうした種類のライブラリをドキュメント無しで公開してもしばしば悪びれず、ライブラリが何をするのかしばしば書かず、ただこれからすべて潰して書き直すつもりだと述べます。悲しいことに、このことが意味するのは Hackage の名前空間の大部分は行き詰まりの少し問題のあるコードで汚されてしまっているということなのです。

多くの他の言語の生態系（Python や NodeJS や Ruby）は前者の哲学を好みますから、そこから Haskell にやってくると**幾千ものライブラリが文書も目的の説明も丸っきりないまま存在する**のを見てゾッとすることでしょう。2つの哲学の文化的な差や、Hackage の現在の文化的な状態の持続可能性は、難しい問題なのです。

言うまでも無く、非常に低品質の Haskell コードや文書は今も巷に溢れており、ライブラリの評価において保守を貫くことは必要な技術なのです。

大まかに言えば、Haddock のドキュメントが**最小限の実用例**を持っていないライブラリは、たいてい「コメント求む」スタイルのライブラリであると考えてよく、おそらく使うのは避けたほうがいいです。

安定で使用可能なライブラリをアップロードしていると考えてよい作者は何人かいます。以下に限られるわけではありませんが、一部を紹介しておきます。

* ブライアン＝オサリバン (Bryan O'Sullivan)
* ジョハン＝ティベル (Johan Tibell)
* サイモン＝マーロー(Simon Marlow)
* ゲーブリエル＝ゴンザレス (Gabriel Gonzalez)
* ローマン＝レシュチンスキー (Roman Leshchinskiy)

## GHCi

GHCi は GHC コンパイラのための対話式シェルです。私たちがほとんどの時間を費やすのが GHCi です。

コマンド | ショートカット | 動作
--- | --- | ---
`:reload` | `:r` | コードの再読み込み
`:type` | `:t` | 型を調べる
`:kind` | `:k` | 種を調べる
`:info` | `:i` | 情報取得
`:print` | `:p` | 式の表示
`:edit` | `:e` | システムのエディタでファイルを読み込む

調べるコマンドが、Haskell のコードに関するデバッグと対話における根幹部分です。

```haskell
λ: :type 3
3 :: Num a => a
```

```haskell
λ: :kind Either
Either :: * -> * -> *
```

```haskell
λ: :info Functor
class Functor f where
  fmap :: (a -> b) -> f a -> f b
  (<$) :: a -> f b -> f a
        -- Defined in `GHC.Base'
  ...
```

```haskell
λ: :i (:)
data [] a = ... | a : [a]       -- Defined in `GHC.Types'
infixr 5 :
```

シェルのグローバルな環境の現在の状態も調べられます。例えば、モジュールレベルの束縛と型：

```haskell
λ: :browse
λ: :show bindings
```

あるいはモジュールレベルのインポートされたもの：

```haskell
λ: :show imports
import Prelude -- implicit
import Data.Eq
import Control.Monad
```

あるいはコンパイラレベルのフラグとプラグマ：

```haskell
λ: :set
options currently set: none.
base language is: Haskell2010
with the following modifiers:
  -XNoDatatypeContexts
  -XNondecreasingIndentation
GHCi-specific dynamic flag settings:
other dynamic, non-language, flag settings:
  -fimplicit-import-qualified
warning settings:

λ: :showi language
base language is: Haskell2010
with the following modifiers:
  -XNoDatatypeContexts
  -XNondecreasingIndentation
  -XExtendedDefaultRules
```

言語拡張とコンパイラのプラグマはプロンプトで設定できます。[Flag
Reference](http://www.haskell.org/ghc/docs/latest/html/users_guide/flag-reference.html)［日本語訳は[フラグ早見表](http://www.kotha.net/ghcguide_ja/latest/flag-reference.html)］を見れば、コンパイラのフラグの非常に多様なオプションを知ることができます。例えば、広く使われているものはこちらです。

```haskell
:set -XNoMonomorphismRestriction
:set -fno-warn-unused-do-bind
```

対話的なオプションのためのコマンドのいくつかにはショートカットがあります。

 | 機能
--- | ---
``+t`` | 評価された式の型を表示する。
``+s`` | 時間の掛かり方とメモリの使用を表示する。
``+m`` | ``:{`` と ``:}`` で囲めば複数行の式を書けるようにする。

```haskell
λ: :set +t
λ: []
[]
it :: [a]
```

```haskell
λ: :set +s
λ: foldr (+) 0 [1..25]
325
it :: Prelude.Integer
(0.02 secs, 4900952 bytes)
```

```haskell
λ: :{
λ:| let foo = do
λ:|           putStrLn "hello ghci"
λ:| :}
λ: foo
"hello ghci"
```

GHCi シェルの設定は ``ghci.conf`` を、``$HOME/.ghc/`` に、あるいはカレントディレクトリに ``./.ghci.conf`` として定義することにより、グローバルにカスタマイズすることができます。

例えば、GHCi 内から Hoogle の型の検索を使用するコマンドを追加することができます。

```bash
cabal install hoogle
```

コマンドを ``ghci.conf`` に追加することで使用できるようになります。

```haskell
:set prompt "λ: "

:def hlint const . return $ ":! hlint \"src\""
:def hoogle \s -> return $ ":! hoogle --count=15 \"" ++ s ++ "\""
```

```haskell
λ: :hoogle (a -> b) -> f a -> f b
Data.Traversable fmapDefault :: Traversable t => (a -> b) -> t a -> t b
Prelude fmap :: Functor f => (a -> b) -> f a -> f b
```

華麗にキメるには、そういう生き方をしたいのならば、GHC のプロンプトを ``λ`` や ``ΠΣ`` にするのが良いでしょう。

```haskell
:set prompt "λ: "
:set prompt "ΠΣ: "
```

## エディタの統合

Haskell には、部分式の型を調べたり、文法チェックをしたり、型検査をしたり、コードの補完をしたりといった、対話式の開発上のフィードバックや機能を得るのに使えるエディタのツールが、いろいろあります。

![](https://raw.githubusercontent.com/sdiehl/wiwinwlh/master/img/errors.png)

Haskell での開発のためのプログラマのエディタの様々な設定を素早く行うための、元々パッケージ化されている設定が、数多く存在します。

**Vim**

[haskell-vim-now](https://github.com/begriffs/haskell-vim-now)

**Emacs**

[emacs-haskell-config](https://github.com/chrisdone/emacs-haskell-config)

これらのパッケージの多くが裏で使っているツールは、たいてい cabal で手に入ります。

```haskell
cabal install hdevtools
cabal install ghc-mod
cabal install hlint
cabal install ghcid
cabal install ghci-ng
```

特に、``ghc-mod`` と ``hdevtools`` は、効率と開発性を著しく向上させます。

参照：

* [A Vim + Haskell Workflow](http://www.stephendiehl.com/posts/vim_haskell.html)

## ボトム

```haskell
error :: String -> a
undefined :: a
```

ボトムは、全ての型が持っている特別な値です。評価されれば、Haskell の意味論は意味のある値を返さなくなります。しばしば ⊥ という記号で書かれます（コンパイラがあなたを引っくり返しているのだと思いましょう）。

無限ループの項の例：

```haskell
f :: a
f = let x = x in x
```

``undefined`` 関数は、しかしながら、不完全なプログラムを書くのに辻褄を合わせたり、デバッグしたりするときに、恐ろしく実用的です。

```haskell
f :: a -> Complicated Type
f = undefined -- 明日書こう、今日は型検査しよう！
```

非網羅的パターンマッチングによる部分関数は、おそらくボトムを生み出す原因として最もありふれたものでしょう。

```haskell
data F = A | B
case x of
  A -> ()
```

上記のコードは、非網羅的パターンのための例外が加えられた下記の GHC のコアに翻訳されます。``-fwarn-incomplete-patterns`` や ``-fwarn-incomplete-uni-patterns`` フラグを使えば、GHC は不完全なパターンについてより口うるさくなります。

```haskell
case x of _ {
  A -> ();
  B -> patError "<interactive>:3:11-31|case"
}
```

同じことが、一部のフィールドの欠けたレコードの構成についても言えます。ただし、フィールドの欠けた状態でレコードを構成するのはほぼ確実に意味のないことなので、GHC はデフォルトで警告を出します。

```haskell
data Foo = Foo { example1 :: Int }
f = Foo {}
```

またしても、コンパイラによりエラーの項が追加されます。

```haskell
Foo (recConError "<interactive>:4:9-12|a")
```

直ちには明らかでないですが、これらは Prelude 全体で大々的に使われています。それには実用上の理由も歴史上の理由もあります。標準的な例として、``head`` は ``[a] -> a`` と書くと、ボトム無しには適切に型付けできません。

```haskell
import GHC.Err
import Prelude hiding (head, (!!), undefined)

-- 脱生成関数

undefined :: a
undefined =  error "Prelude.undefined"

head :: [a] -> a
head (x:_) =  x
head []    =  error "Prelude.head: empty list"

(!!) :: [a] -> Int -> a
xs     !! n | n < 0 =  error "Prelude.!!: negative index"
[]     !! _         =  error "Prelude.!!: index too large"
(x:_)  !! 0         =  x
(_:xs) !! n         =  xs !! (n-1)
```

これらの部分関数が製品のコードで雑に乱用されているのを見ることはあまり無く、より好まれているのは、代わりに ``Data.Maybe`` で提供されている安全な変種を、通常の折り畳み関数である ``maybe`` や ``either`` と共に用いたり、パターンマッチングを使ったりするという手法です。

```haskell
listToMaybe :: [a] -> Maybe a
listToMaybe []     =  Nothing
listToMaybe (a:_)  =  Just a
```

エラーとして定義されたボトムは呼び出されても普通は位置情報を生成しませんが、アサーションを提供するための関数 ``assert`` を使えば、``undefined`` や ``error`` が呼び出された場所の位置情報をサラッと生成することができます。

```haskell
import GHC.Base

foo :: a
foo = undefined
-- *** Exception: Prelude.undefined

bar :: a
bar = assert False undefined
-- *** Exception: src/fail.hs:8:7-12: Assertion failed
```

参照：

* [Avoiding Partial Functions](http://www.haskell.org/haskellwiki/Avoiding_partial_functions)

## 網羅性

Haskell のパターンマッチングは非網羅的なパターン、即ち網羅的でなく、値を生まずに発散する場合分けを許しています。

非網羅性の生む部分関数には賛否両論があり、非網羅的パターンを各所で使っているのは危険なコードの兆候だとされています。とはいえ、言語から非網羅的パターンをすべて、徹底的に取り除いてしまうと、あまりにも窮屈になり、余りにも多くの問題のないプログラムを禁止してしまいますが。

例えば、以下の関数は ``Nothing`` を与えられると実行時にクラッシュするが、そのことを除けばきちんと型検査が通るプログラムです。

```haskell
unsafe (Just x) = x + 1
```

しかし、そういうことを警告させたり、部分的、あるいは全域定期に完全に禁止させたりできる、コンパイラに対するフラグがあります。

```haskell
$ ghc -c -Wall -Werror A.hs
A.hs:3:1:
    Warning: Pattern match(es) are non-exhaustive
             In an equation for `unsafe': Patterns not matched: Nothing
```

``-Wall`` や不完全パターンに対するフラグは、各モジュールに対して別個に ``OPTIONS_GHC`` プラグマを使って追加することもできます。

```haskell
{-# OPTIONS_GHC -Wall #-}
{-# OPTIONS_GHC -fwarn-incomplete-patterns #-}
```

もっと微妙なのは、ラムダ式の単一の”ユニパターン (uni-pattern)”における暗黙のパターンマッチングです。以下の場合、``Nothing`` を与えられると失敗します。

```haskell
boom = \(Just a) -> something
```

これはしばしば ``let`` や ``do`` のブロックで生じます。これらは脱糖衣により上記の例のようにラムダ式に翻訳されます。

```haskell
boom = let
  Just a = something

boom = do
  Just a <- something
```

``-fwarn-incomplete-uni-patterns`` フラグを付ければ、GHC はこれらについても警告してくれます。

大まかに言って、非自明なプログラムはある程度は部分関数を使うものであり、それは紛れもない事実なのです。要は、プログラマには Haskell の型システムで明示されない義務が存在するのです。LiquidHaskell のような将来のプロジェクトでは、より洗練された細別型 (refinement type) を使ってこれを解決する方法が得られるかもしれません。とはいえ、それも研究上で未解決の難問なのですが。

## デバッガ

使われることは割と少ないですが、GHCi には実は組み込みのデバッガがあります。ボトムによる捕捉されない例外や非同期による例外をデバッグするのは、gdb でセグメンテーション違反をデバッグするのと同様の流儀です。

```haskell
λ: :set -fbreak-on-exception
λ: :trace main
λ: :hist
λ: :back
```

## スタックトレース

また、実行時のプロファイリングを有効にすると、GHC は発散するボトムの項（``error`` や ``undefined``）に当たると、スタックトレースを表示してくれます。ただしこれには特別なフラグとプロファイリングを有効にする必要があり、両方ともデフォルトでは無効になっています。例えば、

```haskell
import Control.Exception

f x = g x

g x = error (show x)

main = try (evaluate (f ())) :: IO (Either SomeException ())
```

```haskell
$ ghc -O0 -rtsopts=all -prof -auto-all --make stacktrace.hs
./stacktrace +RTS -xc
```

こうすると本当に実行時に、関数 ``g`` で例外が起こったと教えてくれて、呼び出しスタックを列挙してくれます。

```haskell
*** Exception (reporting due to +RTS -xc): (THUNK_2_0), stack trace:
  Main.g,
  called from Main.f,
  called from Main.main,
  called from Main.CAF
  --> evaluated by: Main.main,
  called from Main.CAF
```

元々の呼び出しスタックをソースコードにあるままの状態で保つために、``-O0`` を付けて、最適化が為されないようにして実行するのが良いです。GHC はプログラムをかなり大幅に変更するので、最適化がかかると、呼び出しスタックはしばしば全く違うものになるのです。

参照：

* [xc flag](https://downloads.haskell.org/~ghc/latest/docs/html/users_guide/runtime-control.html#idp13041968)

## トレース

Haskell は純粋なので、ほとんどのコードがそれ自体で独立して取り出して考えることが出来る、という独特な性質があります。たとえば、多くの場合、"printf" スタイルのデバッグをしなくとも、単に GHCi を開いて関数をテストすればいいのです。しかしそれでも、Haskell には危険な ``trace`` 関数があります。この関数は任意の ``print`` 文を IO モナド外で動かすのに使えます。

```haskell
import Debug.Trace

example1 :: Int
example1 = trace "impure print" 1

example2 :: Int
example2 = traceShow "tracing" 2

example3 :: [Int]
example3 = [trace "will not be called" 3]

main :: IO ()
main = do
  print example1
  print example2
  print $ length example3
-- impure print
-- 1
-- "tracing"
-- 2
-- 1
```

この関数自体が非純粋であり（裏で ``unsafePerformIO`` を使っています）、安定したコードにしたければ使うべきではありません。

``trace`` 関数に限らず、いくつかのモナディックなパターンがかなり広く使われています。

```haskell
import Text.Printf
import Debug.Trace

traceM :: (Monad m) => String -> m ()
traceM string = trace string $ return ()

traceShowM :: (Show a, Monad m) => a -> m ()
traceShowM = traceM . show

tracePrintfM :: (Monad m, PrintfArg a) => String -> a -> m ()
tracePrintfM s = traceM . printf s
```

## 型の穴

GHC 7.8 以来、新たに、**型の穴 (typed holes)** を用いて不完全なプログラムをデバッグすることが出来るようになりました。宣言の右辺の好きな値をアンダースコアに置き換えると、GHC は型検査の間に、プログラムの型検査を通すためにプログラムのその場所に埋めるべき値を示すエラーを投げます。

```haskell
instance Functor [] where
  fmap f (x:xs) = f x : fmap f _
```

```
[1 of 1] Compiling Main             ( src/typedholes.hs, interpreted )

src/typedholes.hs:7:32:
    Found hole ‘_’ with type: [a]
    Where: ‘a’ is a rigid type variable bound by
               the type signature for fmap :: (a -> b) -> [a] -> [b]
               at src/typedholes.hs:7:3
    Relevant bindings include
      xs :: [a] (bound at src/typedholes.hs:7:13)
      x :: a (bound at src/typedholes.hs:7:11)
      f :: a -> b (bound at src/typedholes.hs:7:8)
      fmap :: (a -> b) -> [a] -> [b] (bound at src/typedholes.hs:7:3)
    In the second argument of ‘fmap’, namely ‘_’
    In the second argument of ‘(:)’, namely ‘fmap f _’
    In the expression: f x : fmap f _
Failed, modules loaded: none.
```

GHC は、プログラムを完成させるのに必要な式は ``xs :: [a]`` であるということを正しく示しています。

## Nix

Nix は cabal よりも扱う範囲の大きいパッケージ管理システムです。一般には Haskell に特化したプロジェクトではありませんが、既存の cabal の基本構造に合うように、多くの労力が費やされてきました。**Nix は cabal の代わりとなるものではありません**が、Nix を使えば cabal の仕事の一部を行うことができます。Haskell のライブラリ（バイナリのパッケージからインストールされたもの）と、コンパイルされた Haskell のプログラムにリンクさせられる任意のシステムライブラリとを含められる、独立した開発環境を築けるのです。

Nix を使うべきかどうかという問題は、ある側面ではいくらか意見が分かれます。Nix を使うには、より大きいシステムにわざわざ入り、Nix の全く異なる仕様言語で、設定ファイルを余計にいくつか書かねばならないからです。Haskell と Nix が将来どうなるかは分かりませんし、Nix が cabal の現在の問題点を回避する次善策に過ぎないのか、全体を高度にまとめるモデルなのかも分かりません。

NixOS パッケージマネージャーがインストールされれば、その場で、nix シェルを、nixos レポジトリにインストールされたいくつかのパッケージがある状態で、使い始められます。

```bash
$ nix-shell -p haskellPackages.parsec -p haskellPackages.mtl --command ghci
```

もちろんこれは Haskell のパッケージに限られたことではなく、幅広いバイナリのパッケージやライブラリを入手することができます。ライブラリが GNU readline の特定のバージョンに依存しているならば、Nix はたとえばこの依存関係を上手く管理できます。システムライブラリが ``cabal-install`` のスコープの外にあるのとは対照的です。

```bash
$ nix-shell -p llvm -p julia -p emacs
```

Haskell に対する Nix の作業の流れは、以下の通りです。

```bash
$ cabal init
... usual setup ...
$ cabal2nix mylibrary.cabal --sha256=0 > shell.nix
```

これにより、以下のようなファイルが生成されます。

```ocaml
# This file was auto-generated by cabal2nix. Please do NOT edit manually!

{ cabal, mtl, transformers
}:

cabal.mkDerivation (self: {
  pname = "mylibrary";
  version = "0.1.0.0";
  sha256 = "0";
  isLibrary = true;
  isExecutable = true;
  buildDepends = [
    mtl transformers
  ];
})
```

このファイルを手動で編集する必要があります。

```ocaml
# This file was auto-generated by cabal2nix. Please do NOT edit manually!

{ haskellPackages ? (import <nixpkgs> {}).haskellPackages }:

haskellPackages.cabal.mkDerivation (self: {
  pname = "mylibrary";
  version = "0.1.0.0";
  src = "./.";
  isLibrary = true;
  isExecutable = true;
  buildDepends = with haskellPackages; [
    mtl transformers
  ];
  buildTools = with haskellPackages; [ cabalInstall ];
})
```

いいですね。これで、cabal の REPL を自分のプロジェクトのために起動させることができます。

```bash
$ nix-shell --command "cabal repl"
```

この過程は cabal2nix4dev という別のライブラリが自動化しています。

参照：

* [cabal2nix4dev](https://github.com/dave4420/cabal2nix4dev/blob/master/cabal2nix4dev)

## Haddock

Haddock は Haskell のソースコードに対する説明文書を自動で作成するツールです。通常の cabal のツールチェーンに統合されています。

```haskell
-- | f の説明
f :: a -> a
f = ...
```

```haskell
-- | 複数の引数を持つ関数 fmap に対する
-- 複数行の説明
fmap :: Functor f =>
     => (a -> b)  -- ^ 関数
     -> f a       -- ^ 入力
     -> f b       -- ^ 出力
```

```haskell
data T a b
  = A a -- ^ A の説明
  | B b -- ^ B の説明
```

モジュールの内部の要素（値、型、クラス）は、シングルクオートで識別子を囲むことでハイパーリンクを貼ることができます。

```haskell
data T a b
  = A a -- ^ 'A' の説明
  | B b -- ^ 'B' の説明
```

モジュール自体は、ダブルクオートで囲むことで参照できます。

```haskell
-- | ここでは "Data.Text" を使っていて、
-- 'Data.Text.pack' 関数をインポートしている。
```

```haskell
-- | コードブロックの例
--
-- @
--    f x = f (f x)
-- @

-- > f x = f (f x)
```

```haskell
-- | 対話式のシェルセッションの例
--
-- >>> factorial 5
-- 120
```

特定のブロックに対して、モジュールのブロックのコメントの先頭に星印を付けることで、ヘッダを付けることができます。

```haskell
module Foo (
  -- * ヘッダ
  example1,
  example2
)
```

説明文を参照している ``$`` ブロックをモジュールの本体に付けることで、セクションに対して詳しい説明を与えることもできます。

```haskell
module Foo (
  -- $section1
  example1,
  example2
)

-- $section1
-- これは、シンボル 'example1' と 'example2' を
-- 説明している文書です。
```

以下の構文によりリンクを加えることができます。

```
<URL 文字列>
```

画像も含められます。ただし、パスは Haddock からの相対パスか絶対パスでなければなりません。

```
<<図.png タイトル>>
```

ソースのプラグマにより、モジュールレベルでもプロジェクトレベルでも、Haddock のオプションを指定できます。

```haskell
{-# OPTIONS_HADDOCK show-extensions, ignore-exports #-}
```

オプション | 説明
--- | ---
ignore-exports | エクスポートリストを無視して、スコープにあるすべてのシグニチャを含める。
not-home | モジュールは最上層にある文書とはされない。
show-extensions | 使用している言語拡張を文書で表示する。
hide | モジュールを Haddock から無理やり隠す。
prune | 説明のない定義を省く。
