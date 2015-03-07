# <a name="Prelude">Prelude</a>

## 目次

* [使わない方がいいのはどれ？](#what-to-avoid)
* [部分関数](#partial-functions)
* [safe](#safe)
* [ブール値の不透明性](#boolean-blindness)
* [Foldable と Traversable](#foldable-traversable)
* [余再帰](#corecursion)
* [split](#split)
* [monad-loops](#monad-loops)

## <a name="what-to-avoid">使わない方がいいのはどれ？</a>

Haskell は 25 歳の言語なので、関数的プログラムを組み立てて合成する方法も何度か革命的な変化を受けました。しかし、その結果 Prelude のいろいろな部分が古い流儀の考え方を未だに反映していますが、Haskell の生態系で重要な部分を壊さずにそれらを取り除くことはできないのです。

今のところ、Prelude のどの部分を使うべきでどの部分を使うべきでないのかという情報は、本当に民間伝承にしか存在しません。ほとんどの入門書はこの話題に触れず、単純さのために Prelude を広範囲で利用します。

Prelude についての短い助言は：

* ``map`` の代わりに ``fmap`` を使いましょう。
* Control.Monad や Data.List にあるバージョンの走査ではなく、Foldable や Traversable を使いましょう。
* ``head`` や ``read`` 等の部分関数はなるべく使わず、全域関数の変種を使いましょう。
* 非同期の例外は使わないようにしましょう。
* ブール値の不透明な関数を使うのは避けましょう。

リスト型の Foldable のインスタンスはしばしば Prelude に歴史的理由で残されている単相のバージョンと衝突します。ですから多くの場合、これらの関数を暗黙のインポートから明示的に排除し、Foldable や Traversable を使わせることが望ましいです。

```haskell
import  Data.List hiding (
    all , and , any , concat , concatMap find , foldl ,
    foldl' , foldl1 , foldr , foldr1 , mapAccumL ,
    mapAccumR , maximum , maximumBy , minimum ,
    minimumBy , notElem , or , product , sum )

import Control.Monad hiding (
    forM , forM_ , mapM , mapM_ , msum , sequence , sequence_ )
```

もちろん、Prelude を単に明示的に使用したいだけであることも多く、Prelude を修飾子を付けて (qualified) 名前空間全体を暗黙にインポートすることなく、使いたい時だけ使うこともできます。

```haskell
import qualified Prelude as P
```

しかしこうしても、インポートが明示的か暗黙的かに関わらず、いくつかの型クラスのインスタンスと型クラスを持ち込んでしまいます。Prelude にあるものを本当に何も使いたくなければ、（組み込みのクラスのインスタンスを除いて）Prelude 全体を除外するという選択肢もあります。それには ``-XNoImplicitPrelude`` プラグマを使います。

```haskell
{-# LANGUAGE NoImplicitPrelude #-}
```

プロジェクト全体が暗黙の Prelude 無しにコンパイルされると仮定すると、Prelude 自体を完全に複製することもできます。同じ機能の大部分を、より現代的な設計原理に合うような方法で提供するパッケージが、いくつか現れています。

* [base-prelude](http://hackage.haskell.org/package/base-prelude)
* [basic-prelude](http://hackage.haskell.org/package/basic-prelude)
* [classy-prelude](http://hackage.haskell.org/package/classy-prelude)

## <a name="partial-functions">部分関数</a>

**部分関数**は、与えられた入力に対し常に停止し値を生むとは限らない関数である。反対に、**全域関数**は全ての入力に対し停止し常に定義されている。以前述べたように、Prelude の歴史的な特定の部分は完全に部分関数です。

部分関数と全域関数の違いは、コンパイラが部分関数の実行時の安全性を言語の上で明示された情報だけから推論することができず、安全性の証明自体はユーザーに保証する責任があるということです。ユーザーが無効な入力は生まれないということを保証できる場合には、部分関数は安全に使用できます。しかし、未検査の性質一般に言えることですが、安全か危険かはプログラマの熱心さに依存するものです。これは Haskell の全体的な哲学の真逆を行くものであり、必要でなければ部分関数は使わない方がよいのです。

```haskell
head :: [a] -> a
read :: Read a => String -> a
(!!) :: [a] -> Int -> a
```

## <a name="safe">safe</a>

Prelude に歴史的な部分関数の全域な変種（``Text.Read.readMaybe`` など）がある場合もありますが、しばしばそうした変種は ``safe`` などのさまざまな多目的ライブラリで見つかります。

ここで提供されている全域のバージョンは 3 つのグループに分類されます。

* ``May``：関数が入力で定義されていなければ Nothing を返します。
* ``Def``：関数が入力で定義されていなければデフォルトの値を返します。
* ``Note``：関数が入力で定義されていなければ好きなエラーメッセージで ``error`` を呼び出します。安全ではないですが、ちょっとはデバッグしやすいです！

```haskell
-- Total
headMay :: [a] -> Maybe a
readMay :: Read a => String -> Maybe a
atMay :: [a] -> Int -> Maybe a

-- Total
headDef :: a -> [a] -> a
readDef :: Read a => a -> String -> a
atDef   :: a -> [a] -> Int -> a

-- Partial
headNote :: String -> [a] -> a
readNote :: Read a => String -> String -> a
atNote   :: String -> [a] -> Int -> a
```

## <a name="boolean-blindness">ブール値の不透明性</a>

```haskell
data Bool = True | False

isJust :: Maybe a -> Bool
isJust (Just x) = True
isJust Nothing  = False
```

ブール型の問題は、型レベルでは True と False の間に実質的に何の違いも無いということです。値を受け取って Bool を返す命題は、任意の情報を受け取って破壊してしまうのです。振る舞いについて推論するためには、ブール値の答えを与えた命題の情報源を辿らねばならず、誤った解釈をしてしまう可能性をうんと上げてしまうのです。最悪の場合、関数の仕様が安全か危険か判断するための方法は、述語の名前の文字が情報源を表すと信じることしかないのです！

例えば、各分岐がヌル値の存在下で計算を安全に実行できるかどうかを表す、Bool の値を返す命題をテストすると、得てして予期せぬやりとりが生じます。C や Python のような言語で値がヌルかどうかテストすることは、値が**ヌルで無い**かどうかテストすることと区別が付かない、ということを考えましょう。以下のプログラムのどちらが安全な使用法を表していて、どちらがセグメンテーション違反を起こすでしょうか？

```python
# こっち？
if p(x):
    # x を使う
elif not p(x):
    # x を使わない

# それともこっち？
if p(x):
    # x を使わない
elif not p(x):
    # x を使う
```

詳しく調べようとしても、
For inspection we can't tell without knowing how p is defined, the compiler
can't distinguish the two either and thus the language won't save us if we
happen to mix them up. Instead of making invalid states *unrepresentable* we've
made the invalid state *indistinguishable* from the valid one!

The more desirable practice is to match match on terms which explicitly witness
the proposition as a type ( often in a sum type ) and won't typecheck otherwise.

```haskell
case x of
  Just a  -> use x
  Nothing -> dont use x

-- not ideal
case p x of
  True  -> use x
  False -> dont use x

-- not ideal
if p x
  then use x
  else don't use x
```

To be fair though, many popular languages completely lack the notion of sum types ( the source of many woes in
my opinion ) and only have product types, so this type of reasoning sometimes has no direct equivalence for
those not familiar with ML family languages.

In Haskell, the Prelude provides functions like ``isJust`` and ``fromJust`` both of which can be used to
subvert this kind of reasoning and make it easy to introduce bugs and should often be avoided.

## <a name="foldable-traversable">Foldable と Traversable</a>

If coming from an imperative background retraining one's self to think about iteration over lists in terms of
maps, folds, and scans can be challenging.

```haskell
Prelude.foldl :: (a -> b -> a) -> a -> [b] -> a
Prelude.foldr :: (a -> b -> b) -> b -> [a] -> b

-- pseudocode
foldr f z [a...] = f a (f b ( ... (f y z) ... ))
foldl f z [a...] = f ... (f (f z a) b) ... y
```

For a concrete consider the simple arithmetic sequence over the binary operator
``(+)``:

```haskell
-- foldr (+) 1 [2..]
(1 + (2 + (3 + (4 + ...))))
```

```haskell
-- foldl (+) 1 [2..]
((((1 + 2) + 3) + 4) + ...)
```

Foldable and Traversable are the general interface for all traversals and folds
of any data structure which is parameterized over its element type ( List, Map,
Set, Maybe, ...). These are two classes are used everywhere in modern Haskell
and are extremely important.

A foldable instance allows us to apply functions to data types of monoidal
values that collapse the structure using some logic over ``mappend``.

A traversable instance allows us to apply functions to data types that walk the
structure left-to-right within an applicative context.

```haskell
class (Functor f, Foldable f) => Traversable f where
  traverse :: Applicative g => f (g a) -> g (f a)

class Foldable f where
  foldMap :: Monoid m => (a -> m) -> f a -> m
```

The ``foldMap`` function is extremely general and non-intuitively many of the
monomorphic list folds can themselves be written in terms of this single
polymorphic function.

``foldMap`` takes a function of values to a monoidal quantity, a functor over
the values and collapses the functor into the monoid. For instance for the
trivial Sum monoid.

```haskell
λ: foldMap Sum [1..10]
Sum {getSum = 55}
```

The full Foldable class (with all default implementations) contains a variety of
derived functions which themselves can be written in terms of ``foldMap`` and
``Endo``.

```haskell
newtype Endo a = Endo {appEndo :: a -> a}

instance Monoid (Endo a) where
        mempty = Endo id
        Endo f `mappend` Endo g = Endo (f . g)
```

```haskell
class Foldable t where
    fold    :: Monoid m => t m -> m
    foldMap :: Monoid m => (a -> m) -> t a -> m

    foldr   :: (a -> b -> b) -> b -> t a -> b
    foldr'  :: (a -> b -> b) -> b -> t a -> b

    foldl   :: (b -> a -> b) -> b -> t a -> b
    foldl'  :: (b -> a -> b) -> b -> t a -> b

    foldr1  :: (a -> a -> a) -> t a -> a
    foldl1  :: (a -> a -> a) -> t a -> a
```

For example:

```haskell
foldr :: (a -> b -> b) -> b -> t a -> b
foldr f z t = appEndo (foldMap (Endo . f) t) z
```

Most of the operations over lists can be generalized in terms in combinations of
Foldable and Traversable to derive more general functions that work over all
data structures implementing Foldable.

```haskell
Data.Foldable.elem    :: (Eq a, Foldable t) => a -> t a -> Bool
Data.Foldable.sum     :: (Num a, Foldable t) => t a -> a
Data.Foldable.minimum :: (Ord a, Foldable t) => t a -> a
Data.Traversable.mapM :: (Monad m, Traversable t) => (a -> m b) -> t a -> m (t b)
```

Unfortunately for historical reasons the names exported by foldable quite often conflict with ones defined in
the Prelude, either import them qualified or just disable the Prelude. The operations in the Foldable all
specialize to the same behave the same as the ones Prelude for List types.

```haskell
import Data.Monoid
import Data.Foldable
import Data.Traversable

import Control.Applicative
import Control.Monad.Identity (runIdentity)
import Prelude hiding (mapM_, foldr)

-- Rose Tree
data Tree a = Node a [Tree a] deriving (Show)

instance Functor Tree where
  fmap f (Node x ts) = Node (f x) (fmap (fmap f) ts)

instance Traversable Tree where
  traverse f (Node x ts) = Node <$> f x <*> traverse (traverse f) ts

instance Foldable Tree where
  foldMap f (Node x ts) = f x `mappend` foldMap (foldMap f) ts


tree :: Tree Integer
tree = Node 1 [Node 1 [], Node 2 [] ,Node 3 []]


example1 :: IO ()
example1 = mapM_ print tree

example2 :: Integer
example2 = foldr (+) 0 tree

example3 :: Maybe (Tree Integer)
example3 = traverse (\x -> if x > 2 then Just x else Nothing) tree

example4 :: Tree Integer
example4 = runIdentity $ traverse (\x -> pure (x+1)) tree
```

The instances we defined above can also be automatically derived by GHC using several language extensions. The
automatic instances are identical to the hand-written versions above.

```haskell
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveTraversable #-}

data Tree a = Node a [Tree a]
  deriving (Show, Functor, Foldable, Traversable)
```

See: [Typeclassopedia](http://www.haskell.org/haskellwiki/Typeclassopedia)

## <a name="corecursion">余再帰</a>

```haskell
unfoldr :: (b -> Maybe (a, b)) -> b -> [a]
```

A recursive function consumes data and eventually terminates, a corecursive
function generates data and **coterminates**. A corecursive function is said to
*productive* if can always evaluate more of the resulting value in bounded time.

```haskell
import Data.List

f :: Int -> Maybe (Int, Int)
f 0 = Nothing
f x = Just (x, x-1)

rev :: [Int]
rev = unfoldr f 10

fibs :: [Int]
fibs = unfoldr (\(a,b) -> Just (a,(b,a+b))) (0,1)
```

## <a name="split">split</a>

The [split](http://hackage.haskell.org/package/split-0.1.1/docs/Data-List-Split.html) package provides a
variety of missing functions for splitting list and string types.

```haskell
import Data.List.Split

example1 :: [String]
example1 = splitOn "." "foo.bar.baz"
-- ["foo","bar","baz"]

example2 :: [String]
example2 = chunksOf 10 "To be or not to be that is the question."
-- ["To be or n","ot to be t","hat is the"," question."]
```

## <a name="monad-loops">monad-loops</a>

The [monad-loops](http://hackage.haskell.org/package/monad-loops-0.4.2/docs/Control-Monad-Loops.html) package
provides a variety of missing functions for control logic in monadic contexts.


```haskell
whileM :: Monad m => m Bool -> m a -> m [a]
untilM :: Monad m => m a -> m Bool -> m [a]
iterateUntilM :: Monad m => (a -> Bool) -> (a -> m a) -> a -> m a
whileJust :: Monad m => m (Maybe a) -> (a -> m b) -> m [b]
```
