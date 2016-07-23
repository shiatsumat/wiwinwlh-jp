# 遅延性

## はじめに

また、**多くの**インクが費やされた話題です。Haskell の国では今でも、遅延評価 (lazy evaluation) と正格評価 (strict evaluation) との歩み寄りについて議論が交わされていて、どちらかのパラダイムをデフォルトにしようと繊細な主張がなされています。Haskell は複合的なアプローチを取っていて、必要な時には正格評価を行い、デフォルトでは遅延評価を使っています。言うまでもなく、正格評価が遅延評価より悪い振る舞いをする例も、その逆である例も、常に見つかります。

遅延評価を大規模に用いる主な利点は、非有界なデータ構造と有界なデータ構造のそれぞれを操作するアルゴリズムが同じ型シグニチャを持てて、論理を再構成したり、途中の計算を強制評価したりという余計な手間無しに両者を合成できる、ということです。正格評価モデルに遅延性を埋め込もうとする言語はしばしば、アルゴリズムを、非有界な構造を消費するのに便利なものと、有界な構造を操作するものとに分けることになります。正格な言語では、遅延処理と正格処理を混ぜて合わせようとすると、しばしばメモリ上に大きい中間構造を抱える必要が生じます。怠惰 (lazy) な言語ではそうした合成は”普通に動く”のです。

Haskell が言語設計の領域で、工業的に使えるギリギリのところまで、この問題を探究している唯一の言語であるおかげで、こんなことになっているのです。遅延評価に関する知識は、プログラマの集団全体の意識にはあまり浸透しておらず、初心者にとってはしばしば非直感的に感じられます。怠惰なコンパイラの最適化についての更なる教材と研究が必要であるという単純な理由から、Haskell はモデル自体についても熟考しています。

Haskell のパラドックスは、独特であると説明できる多くの概念（遅延性、純粋性、型クラス）を持つがために、実装全体の構造からそれらのうちのどれを切り離すことも難しい、ということです。

参照：

* [Oh My Laziness!](http://alpmestan.com/2013/10/02/oh-my-laziness/)
* [Reasoning about Laziness](http://www.slideshare.net/tibbe/reasoning-about-laziness)
* [Lazy Evaluation of Haskell](http://www.vex.net/~trebla/haskell/lazy.xhtml)
* [More Points For Lazy Evaluation](http://augustss.blogspot.hu/2011/05/more-points-for-lazy-evaluation-in.html)
* [How Lazy Evaluation Works in Haskell](https://hackhands.com/lazy-evaluation-works-haskell/)

## 正格性

ラムダ計算の評価モデルはいくつか存在します。

* 正格：関数本体を見る前に全ての引数を評価する場合、評価は正格であるといいます。
* 非正格：関数本体を見る前に引数を評価しなくてもいい場合、評価は非正格であるといいます。

これらの概念はいくつかのモデルを生みますが、Haskell においては、*必要呼び*モデルを使っています。

モデル | 正格性 | 説明
:---: | :---: | ---
値呼び | 正格 | 関数を見る前に引数を評価する
名前呼び | 非正格 | 引数は評価されない
必要呼び | 非正格 | 引数は評価されないが、式は一度だけ評価される（共有）

## seqとWHNF

いちばん外側のコンストラクタやラムダがこれ以上簡約できない場合、項は**弱頭部正規形** (weak head normal-form, WHNF) であると言います。完全に評価されていて、中に入っているすべての部分式とサンクが評価されている場合、**正規形** (normal form) であると言います。

```haskell
-- 正規形
42
(2, "foo")
\x -> x + 1

-- 正規形でない
1 + 2
(\x -> x + 1) 2
"foo" ++ "bar"
(1 + 1, "foo")

-- 弱頭部正規形
(1 + 1, "foo")
\x -> 2 + 2
'f' : ("oo" ++ "bar")

-- 弱頭部正規形でない
1 + 1
(\x -> x + 1) 2
"foo" ++ "bar"
```

Haskell では、通常の評価はコアの case 文の外側のコンストラクタでのみ起こります。リストに対しパターンマッチングをするとき、リストの全ての値を暗黙のうちに強制評価することはありません。データ構造の要素はいちばん外側のコンストラクタが出てくるまでしか評価しません。例えば、リストの長さを評価するには、外側のコンスのみを調べていけばよく、中の値は見なくてよいのです。

```haskell
λ: length [undefined, 1]
2

λ: head [undefined, 1]
Prelude.undefined

λ: snd (undefined, 1)
1

λ: fst (undefined, 1)
Prelude.undefined
```

例えば、怠惰な言語では、発散する項を含んでいるにもかかわらず以下のプログラムは停止します。

```haskell
ignore :: a -> Int
ignore x = 0

loop :: a
loop = loop

main :: IO ()
main = print $ ignore loop
```

OCaml のような正格な言語では（ここでは保留 (suspension) の機能を無視することにして）、同じプログラムは発散します。

```haskell
let ignore x = 0;;
let rec loop a = loop a;;

print_int (ignore (loop ()));
```

Haskell では、**サンク** (thunk) が評価されていない計算を表すために作られます。サンクの評価はサンクの**強制評価** (forcing) と呼ばれています。その結果、**更新**という参照透明な作用が生まれます。これは、サンクのメモリでの表現を計算された値に変えるものです。基本的なアイデアは、サンクは一度しか更新されず、結果として得られる値は順々に参照されても共有される、ということです。

``:sprint`` コマンドを使えば、評価を強制せずに式の内部の未評価のサンクの状態を見ることができます。例えば：

```haskell
λ: let a = [1..] :: [Integer]
λ: let b = map (+ 1) a

λ: :sprint a
a = _
λ: :sprint b
b = _
λ: a !! 4
5
λ: :sprint a
a = 1 : 2 : 3 : 4 : 5 : _
λ: b !! 10
12
λ: :sprint a
a = 1 : 2 : 3 : 4 : 5 : 6 : 7 : 8 : 9 : 10 : 11 : _
λ: :sprint b
b = _ : _ : _ : _ : _ : _ : _ : _ : _ : _ : 12 : _
```

サンクが計算中である間、メモリの表現は**ブラックホール**として知られる特別な形式に置き換えられます。これは、計算が進行中であり、ある計算が計算の完了のために自分自身に依存しているときに近道をすることを許すものです。これの実装には、GHC の実行時の動作の細部の比較的繊細な部分が関わっています。

``seq``［順に (sequentially) 評価する］関数は、1 番目の引数が 2 番目の引数の評価の前に WHNF へと評価されるようにして、2 つの項の評価順序を人工的に作り出します。``seq`` 関数の実装には、GHC の実装の細部が関わっています。

```haskell
seq :: a -> b -> b

⊥ `seq` a = ⊥
a `seq` b = b
```

悪名高い ``foldl`` は、不注意に、コンパイラのいくつかの最適化をかけずに使ったとき、スペースリークを生み出す、ということで知られています。正格な ``foldl'`` という変種は、``seq`` を使ってその問題を解決しています。

```haskell
foldl :: (a -> b -> a) -> a -> [b] -> a
foldl f z [] = z
foldl f z (x:xs) = foldl f (f z x) xs
```

```haskell
foldl' :: (a -> b -> a) -> a -> [b] -> a
foldl' _ z [] = z
foldl' f z (x:xs) = let z' = f z x in z' `seq` foldl' f z' xs
```

実用上は、``-O2`` で得られる正格性分析器とインライナを使えば、呼び出している場所でインライン化可能ならば ``foldl`` の正格な変種が使われることが保証されます。ですから、``foldl'`` を使う必要は多くの場合ありません。

重要な注意をしておきます。GHCi はいかなる最適化も行わずに実行するので、GHCi で動くのが遅いプログラムでも、GHC でコンパイルされるとパフォーマンスの特徴は変わっているかもしれません。

## 正格性注釈

``BangPatterns``［ビックリパターン］拡張を使うと、新しい構文を使って、関数の引数を seq でくるんで強制評価させるようにできます。引数にビックリ演算子を付けると、パターンマッチの実行前に弱頭部正規形へと評価されるように強制することができます。この機能を使えば、サンクの巨大な連鎖を生まないように、特定の引数を再帰を通して評価され続けるようにすることができます。

```haskell
{-# LANGUAGE BangPatterns #-}

sum :: Num a => [a] -> a
sum = go 0
  where
    go !acc (x:xs) = go (acc + x) (go xs)
    go  acc []     = acc
```

これは、以下と実質的に等価なコードへと脱糖衣されます。

```haskell
sum :: Num a => [a] -> a
sum = go 0
  where
    go acc _ | acc `seq` False = undefined
    go acc (x:xs)              = go (acc + x) (go xs)
    go acc []                  = acc
```

``seq`` された引数への関数適用はよく行われることなので、特別な演算子があります。

```haskell
($!) :: (a -> b) -> a -> b
f $! x  = let !vx = x in f vx
```

## deepseq

パフォーマンスのために、データ構造を正規形へと、未評価の項を残さず深く評価する必要が生じることはしばしばあります。``deepseq`` ライブラリはこの仕事をしてくれます。

型クラス ``NFData``（Normal Form Data［正規形データ］）を使えば、``NFData`` を実装する部分型からなる構造の全ての要素を ``seq`` することができます。

```haskell
class NFData a where
  rnf :: a -> ()
  rnf a = a `seq` ()

deepseq :: NFData a => a -> b -> a
($!!) :: (NFData a) => (a -> b) -> a -> b
```

```haskell
instance NFData Int
instance NFData (a -> b)

instance NFData a => NFData (Maybe a) where
    rnf Nothing  = ()
    rnf (Just x) = rnf x

instance NFData a => NFData [a] where
    rnf [] = ()
    rnf (x:xs) = rnf x `seq` rnf xs
```

```haskell
[1, undefined] `seq` ()
-- ()

[1, undefined] `deepseq` ()
-- Prelude.undefined
```

データ構造自体が完全に評価されるよう強制するためには、``deepseq`` の両辺に同じ引数を与えます。

```haskell
force :: NFData a => a
force x = x `deepseq` x
```

## 反駁不可パターン

遅延パターンは、外側のコンストラクタに対するマッチを要求せず、値のアクセッサを遅延的に呼び出します。ボトムがある場合は、外側のパターンマッチでは無く、それぞれの呼び出しの場所で失敗が起こります。

```haskell
f :: (a, b) -> Int
f (a,b) = const 1 a

g :: (a, b) -> Int
g ~(a,b) = const 1 a

-- λ: f undefined
-- *** Exception: Prelude.undefined
-- λ: g undefined
-- 1

j :: Maybe t -> t
j ~(Just x) = x

k :: Maybe t -> t
k (Just x) = x

-- λ: j Nothing
-- *** Exception: src/05-laziness/lazy_patterns.hs:15:1-15: Irrefutable pattern failed for pattern (Just x)
--
-- λ: k Nothing
-- *** Exception: src/05-laziness/lazy_patterns.hs:18:1-14: Non-exhaustive patterns in function k
```

## 道徳的な正しさ

遅延評価に関する但し書きがあります。遅延評価があるせいで、関数についての帰納的な推論を行う場合、必ず関数はボトムを含みうるという事実を考慮せねばならない、ということです。また、関数の帰納的な証明についての主張はそれ自体、ボトムの不在を仮定している場合「いい加減な推論に基づくと」という修飾語を暗に付されているものであるとせねばなりません。

"Fast and Loose reasoning is Morally Correct"［いい加減な推論は道徳的には正しい］という論文でジョン＝ヒューズ (John Hughes) らは、完全な言語で同じ意味論を持つ 2 つの項は部分的な言語でも関連した意味論を持つということを示し、ある提言をしました。その提言に従えば、遅延言語についての証明が実際に厳密で健全なものとなるための、きちんと記述された特定のいくつかの条件があれば、2 つの領域の間で知識を翻訳することができいます。

参照：

* [Fast and Loose Reasoning is Morally Correct](http://www.cse.chalmers.se/~nad/publications/danielsson-et-al-popl2006.html)
