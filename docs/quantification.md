# 量化

## 全称量化

全称量化は、Haskell で多相性を実現する根幹の仕組みです。全称量化の本質は、型の集合に対して同じ方法で操作し、振る舞いが扱っている範囲のすべての型の振る舞い**のみ**により定められるような関数を表現できることです。

```haskell
{-# LANGUAGE ExplicitForAll #-}

-- ∀a. [a]
example1 :: forall a. [a]
example1 = []

-- ∀a. [a]
example2 :: forall a. [a]
example2 = [undefined]

-- ∀a. ∀b. (a → b) → [a] → [b]
map' :: forall a. forall b. (a -> b) -> [a] -> [b]
map' f = foldr ((:) . f) []

-- ∀a. [a] → [a]
reverse' :: forall a. [a] -> [a]
reverse' = foldl (flip (:)) []
```

通常、量化子は型シグニチャでは省略されます。なぜなら、Haskell のありきたりな表層言語では、自由な［束縛されていない］型変数は全称量化されていると仮定しても曖昧さが生じないからです。

## 自由定理

全称量化された型変数は実は、関数の実装についてのかなり多くの深遠な性質を示唆しています。この性質は関数の型シグニチャから導けます。例えば、Haskell の恒等関数は、型シグニチャに対する実装を一つしか持たないことが保証されています。

```haskell
id :: forall a. a -> a
id x = x
```

```haskell
fmap :: Functor f => (a -> b) -> f a -> f b
```

fmap に対する自由定理がこれです。

```haskell
forall f g. fmap f . fmap g = fmap (f . g)
```

参照：

* [Automatic generation of free theorems](http://www-ps.iai.uni-bonn.de/cgi-bin/free-theorems-webui.cgi)

## 型システム

**ヒンドリー・ミルナー型システム**

ヒンドリー・ミルナー型システム (Hindley-Milner type system, HM type system) は、歴史的には、多相性と、常に主要型 (principal type) を決定できるいくつかの型推論の技法を両立させた、初めての型付きラムダ計算の一つとして重要です。

```
e : x
  | λx:t.e            -- 値の抽象
  | e1 e2             -- 適用
  | let x = e1 in e2  -- let 式

t : t -> t     -- 関数型
  | a          -- 型変数

σ : ∀ a . t    -- 型の機構
```

実装では、``generalize``［総称化］という関数が型の中の全ての型変数を多相の型変数に変換し、型のスキーム［機構］を生みます。``instantiate``［実体化］という関数は、スキームを型に対応させますが、さらに任意の多相の変数を、束縛されていない型変数へと変換します。

**任意ランク多相**

System F は Haskell の基盤にある型システムです。System F は、HM の全ての型が System F で表現できるという点で、HM 型システムを含んでいます。System F は、他の文献では **ジラール・レナルズ多相ラムダ計算** (Girald-Reynolds polymorphic lambda calculus) とか**二階ラムダ計算** (second-order lambda calculus) と呼ばれることがあります。

```
t : t -> t     -- 関数型
  | a          -- 型変数
  | ∀ a . t    -- 全称量化

e : x          -- 変数
  | λ(x:t).e   -- 値の抽象
  | e1 e2      -- 値の適用
  | Λa.e       -- 型の抽象
  | e t       -- 型の適用
```

例（コメントは GHC のコアで等価なコード）：

```
id : ∀ t. t -> t
id = Λt. λx:t. x
-- id :: forall t. t -> t
-- id = \ (@ t) (x :: t) -> x

tr : ∀ a. ∀ b. a -> b -> a
tr = Λa. Λb. λx:a. λy:b. x
-- tr :: forall a b. a -> b -> a
-- tr = \ (@ a) (@ b) (x :: a) (y :: b) -> x

fl : ∀ a. ∀ b. a -> b -> b
fl = Λa. Λb. λx:a. λy:b. y
-- fl :: forall a b. a -> b -> b
-- fl = \ (@ a) (@ b) (x :: a) (y :: b) -> y

nil : ∀ a. [a]
nil = Λa. Λb. λz:b. λf:(a -> b -> b). z
-- nil :: forall a. [a]
-- nil = \ (@ a) (@ b) (z :: b) (f :: a -> b -> b) -> z

cons : ∀ a. a -> [a] -> [a]
cons = Λa. λx:a. λxs:(∀ b. b -> (a -> b -> b) -> b).
    Λb. λz:b. λf : (a -> b -> b). f x (xs b z f)
-- cons :: forall a. a
--       -> (forall b. (a -> b -> b) -> b) -> (forall b. (a -> b -> b) -> b)
-- cons = \ (@ a) (x :: a) (xs :: forall b. (a -> b -> b) -> b)
--     (@ b) (z :: b) (f :: a -> b -> b) -> f x (xs @ b z f)
```

通常、Haskellの型検査器は、型式の本体の内部で量化子が現れないように、型変数の全ての全称量化子は最も外側にあると推論します。これを冠頭制限 (prenex restriction) といいます。これは、System F でならば表現可能な型シグネチャを禁止してしまいますが、型推論をずっと簡単にするという利点はあります。

``-XRankNTypes``は冠頭制限をゆるめ、型の本体の内部に明示的に量化子を置くことが出来るようにしてくれます。悪い知らせは、このゆるいシステムでの型推論の一般的な問題は、一般には決定不能であるということです。ですから、RankNTypes を使う関数は明示的に型注釈を付ける必要があり、そうでなければランク 1 と推論されるか、全く型検査が通りません。

```haskell
{-# LANGUAGE RankNTypes #-}

-- Can't unify ( Bool ~ Char )
rank1 :: forall a. (a -> a) -> (Bool, Char)
rank1 f = (f True, f 'a')

rank2 :: (forall a. a -> a) -> (Bool, Char)
rank2 f = (f True, f 'a')

auto :: (forall a. a -> a) -> (forall b. b -> b)
auto x = x

xauto :: forall a. (forall b. b -> b) -> a -> a
xauto f = f
```

```haskell
単相、ランク 0: t
多相、ランク 1: forall a. a -> t
多相、ランク 2: (forall a. a -> t) -> t
多相、ランク 3: ((forall a. a -> t) -> t) -> t
```

重要な付言をしておきます。高ランクの型で明示的な量化子で束縛された型変数は囲まれたスコープから出ることはできません。型検査器はこれをきちんと守るために、高ランクの型の内部で束縛された変数（スコーレム定数と言います）が、自由なメタ型変数と同一化されないように強制しています。

```haskell
{-# LANGUAGE RankNTypes #-}

escape :: (forall a. a -> a) -> Int
escape f = f 0

g x = escape (\a -> x)
```

この例では、式がきちんと型付けされるために ``f`` は ``Int -> Int`` の型を持たねばならず、それゆえ ``a ~ Int`` であることが型全体で要求されていますが、``a`` は量化子の下で束縛されているので、``Int`` と同一化できません。ゆえに、型検査器はスコーレム捕捉のエラーで失敗することになります。

```
Couldn't match expected type `a' with actual type `t'
`a' is a rigid type variable bound by a type expected by the context: a -> a
`t' is a rigid type variable bound by the inferred type of g :: t -> Int
In the expression: x In the first argument of `escape', namely `(\ a -> x)'
In the expression: escape (\ a -> x)
```

この性質は実際には、特定の型変数のスコープや使用について何種類かの不変性を課して、活用することもできます。例えば、ST モナドは、別々の状態スレッドを持つ複数の ST モナド間の参照の捕捉を防ぐために、ランク 2 の型を使っています。``s`` 型変数はランク 2 の型の中に束縛されていて逃げることができず、ST の内部の実装の詳細が漏れ出さないことを静的に保証し、参照透明性を保っているのです。

## 存在量化

全称量化の本質は、**いかなる**型に対しても同じ方法で操作する関数を表現することです。一方、存在量化については、**ある**未知の型に対して走査する関数を表現できます。存在量化を使えば、存在量化の下にある、データ型を操作するが型シグニチャがその情報を隠している関数を使って、異種の複数の値をまとめて扱うことができます。

```haskell
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE RankNTypes #-}

-- ∃ t. (t, t → t, t → String)
data Box = forall a. Box a (a -> a) (a -> String)

boxa :: Box
boxa = Box 1 negate show

boxb :: Box
boxb = Box "foo" reverse show

apply :: Box -> String
apply (Box x f p) = p (f x)

-- ∃ t. Show t => t
data SBox = forall a. Show a => SBox a

boxes :: [SBox]
boxes = [SBox (), SBox 2, SBox "foo"]

showBox :: SBox -> String
showBox (SBox a) = show a

main :: IO ()
main = mapM_ (putStrLn . showBox) boxes
-- ()
-- 2
-- "foo"
```

``SBox`` に対する存在量化により、Show のインターフェースにより純粋に定義されたいくつかの値を集められますが、値についての他の情報は手に入らず、他のいかなる方法でもアクセスしたりアンパックしたりすることはできません。

```haskell
{-# LANGUAGE RankNTypes #-}

-- この関手はライブラリの内部実装を修正したものです。
type Exists a b = forall f. Functor f => (b -> f b) -> (a -> f a)

type Get a b = a -> b
type Set a b = a -> b -> a

example :: Get a b -> Set a b -> Exists a b
example f g l a = fmap (g a) (l (f a))
```

全称量化を使うと、いわゆる「オブジェクト指向パラダイム」と言われるある種の概念を再現することができます。これは、80 年代後半に人気を博した学派で、現代的な等式で表現する方法を使わず、プログラミングの論理を人間らしいと実体と動作へと分解することを試みました。このモデルを Haskell で再現することは、広く アンチパターンであると考えられています。

参照：

* [Haskell Antipattern: Existential Typeclass](http://lukepalmer.wordpress.com/2010/01/24/haskell-antipattern-existential-typeclass/)

## 不可述型

恐ろしく不安定ですが、GHC は不可述型を部分的にサポートしています。この機能を使えば、型変数を多相型で実体化することができます。つまり、この機能により、量化子は矢印型よりも先に来なければならないという制限が緩められ、量化子が型コンストラクタの内部に置かれても構わなくなるのです。

```haskell
-- Can't unify ( Int ~ Char )

revUni :: forall a. Maybe ([a] -> [a]) -> Maybe ([Int], [Char])
revUni (Just g) = Just (g [3], g "hello")
revUni Nothing  = Nothing
```

```haskell
{-# LANGUAGE ImpredicativeTypes #-}

-- 高ランク多相を使っている
f :: (forall a. [a] -> a) -> (Int, Char)
f get = (get [1,2], get ['a', 'b', 'c'])

-- 不可述多相を使っている
g :: Maybe (forall a. [a] -> a) -> (Int, Char)
g Nothing = (0, '0')
g (Just get) = (get [1,2], get ['a','b','c'])
```

この拡張はほとんど使われておらず、``-XImpredicativeTypes`` は根元から壊れているという考えもあります。まあ GHC は、型シグニチャでタイプミスをしてしまった時でも、この拡張を有効にすることをためらうことなく勧めてくるのですがね！

注目すべき雑学を紹介しておきましょう。``($)`` 演算子は GHC に非常に特殊な方法で組み込まれており、``runST`` の不可述な実体が ``($)`` を通して適用されても構わないようになっています。ST モナドに対して使われる場合だけ、``($)`` 演算子を特別扱いしているのです。これがなんだか汚いハックに見えるとしたら、実際そうなのですが、それでも結構便利なハックなのです。

例えば、``($)`` と全く同じ振る舞いをするはずの関数 ``apply`` を定義したとすると、全く同じ定義であるにもかかわらず、多相の実体化についてエラーが生じるのです！

```haskell
{-# LANGUAGE RankNTypes #-}

import Control.Monad.ST

f `apply` x =  f x

foo :: (forall s. ST s a) -> a
foo st = runST $ st

bar :: (forall s. ST s a) -> a
bar st = runST `apply` st
```

```haskell
    Couldn't match expected type `forall s. ST s a'
                with actual type `ST s0 a'
    In the second argument of `apply', namely `st'
    In the expression: runST `apply` st
    In an equation for `bar': bar st = runST `apply` st
```

参照：

* [SPJ Notes on $](https://www.haskell.org/pipermail/glasgow-haskell-users/2010-November/019431.html)

## スコープのある型変数

通常、関数のトップレベルのシグニチャの内部で使われている型変数は、型シグニチャの内部でのみスコープを持ち、関数本体ではスコープを持たず、項や let/where 節に対しては固定のシグニチャです。``-XScopedTypeVariables`` を有効にすると、この制限が弱まり、トップレベルで言及された型変数が、値レベルの関数本体とそこに含まれるすべてのシグニチャの内部にスコープを持つようになります。

```haskell
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE ScopedTypeVariables #-}

poly :: forall a b c. a -> b -> c -> (a, a)
poly x y z = (f x y, f x z)
  where
    -- 2 番目の引数は型推論により全称量化されている
    -- f :: forall t0 t1. t0 -> t1 -> t0
    f x' _ = x'

mono :: forall a b c. a -> b -> c -> (a, a)
mono x y z = (f x y, f x z)
  where
    -- b はスコープの中にあるので、暗黙には全称量化されない
    f :: a -> b -> a
    f x' _ = x'

example :: IO ()
example = do
  x :: [Int] <- readLn
  print x
```
