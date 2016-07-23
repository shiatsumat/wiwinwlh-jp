# 型族

## 多引数型クラス

平凡な Haskell 98 の型クラスの解決は、非常に単純な環境の簡約により行われます。この簡約により、述語間の依存は最小化され、スーパークラスは解決され、型は頭部正規形へと簡約されます。例えば、

```haskell
(Eq [a], Ord [a]) => [a]
==> Ord a => [a]
```

もし単一引数の型クラスが型のある性質（つまりあるクラスに入っているとか入っていないとかいうこと）を表現しているならば、多引数の型クラスは複数の型の間の関係性を表現しています。例えば、型がある型へと変換できるという関係を表現したければ、このようなクラスを使うかもしれません。

```haskell
{-# LANGUAGE MultiParamTypeClasses #-}

import Data.Char

class Convertible a b where
  convert :: a -> b

instance Convertible Int Integer where
  convert = toInteger

instance Convertible Int Char where
  convert = chr

instance Convertible Char Int where
  convert = ord
```

もちろん、今 ``Convertible Int``に対するインスタンスは一意性が無いので、``a`` だけから ``b`` の型を確実に推論するいいやり方はもう無いのです。これを直すには、``a ->
b`` という関数従属 (functional dependency) を追加しましょう。これは GHC に ``a`` は ``b`` の取りうるインスタンスを一意に決定するということを伝えています。ですから、``Int`` に関する ``Integer`` と ``Char`` の両インスタンスは衝突することになります。

```haskell
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}


import Data.Char

class Convertible a b | a -> b where
  convert :: a -> b

instance Convertible Int Char where
  convert = chr

instance Convertible Char Int where
  convert = ord
```

```haskell
Functional dependencies conflict between instance declarations:
  instance Convertible Int Integer
  instance Convertible Int Char
```

こうすると、インスタンスを一意に定める簡単なやり方があるので、多引数型クラスは使いやすく推論しやすくなります。実質的には、関数従属 ``| a -> b`` は、``a`` が同じでも ``b`` が異なる複数の多引数型クラスインスタンスを定義できないということを表しているのです。

```haskell
λ: convert (42 :: Int)
'42'
λ: convert '*'
42
```

もうちょっとややこしいことをしましょう。``UndecidableInstances`` をオンにすると、クラスの制約が頭部よりも構造上小さくならなければならない、という環境の簡約に対する制約が緩和されます。結果として、暗黙の計算が**型クラスのインスタンスの検索の内部で**生じることになります。ペアノ数の型レベルの表現と組み合わせれば、型レベルで基本的な算術を記述できます。

```haskell
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE UndecidableInstances #-}

data Z
data S n

type Zero  = Z
type One   = S Zero
type Two   = S One
type Three = S Two
type Four  = S Three

zero :: Zero
zero = undefined

one :: One
one = undefined

two :: Two
two = undefined

three :: Three
three = undefined

four :: Four
four = undefined

class Eval a where
  eval :: a -> Int

instance Eval Zero where
  eval _ = 0

instance Eval n => Eval (S n) where
  eval m = 1 + eval (prev m)

class Pred a b | a -> b where
  prev :: a -> b

instance Pred Zero Zero where
  prev = undefined

instance Pred (S n) n where
  prev = undefined

class Add a b c | a b -> c where
  add :: a -> b -> c

instance Add Zero a a where
  add = undefined

instance Add a b c => Add (S a) b (S c) where
  add = undefined

f :: Three
f = add one two

g :: S (S (S (S Z)))
g = add two two

h :: Int
h = eval (add three four)
```

もし型クラスの環境が Prolog に似ていると思ったなら、その通りです。環境修飾子 ``(=>)`` を逆向きに読んで、ターンスタイル ``:-`` として見たら、綺麗に同じ等式になります。

```prolog
add(0, A, A).
add(s(A), B, s(C)) :- add(A, B, C).

pred(0, 0).
pred(S(A), A).
```

これは言ってしまえば型クラスの乱用で、うっかりすると停止しなかったりコンパイル時にオーバーフローしたりします。事前にどういうことになるのかきちんと考えずに ``UndecidableInstances`` をオンにするのはやめましょう。

```haskell
<interactive>:1:1:
    Context reduction stack overflow; size = 201
```

## 型族の基本

型族を使えば、型を引数とし、コンパイル時に型検査の間に評価される、引数を指標とする型あるいは値を返す、型上の関数を書けます。型族には 2 種類の形式があります：**データ族**と**型シノニム族**です。

* **型族**は型上の名前のある関数です。
* **データ族**は型を指標とするデータ型です。

まずは**型シノニム族**を見ていきましょう。型族の構成には構文上 2 種類の等価な方法があります。型クラスの内部で定義された**関連**型族として宣言する方法と、トップレベルで独立して宣言する方法です。以下の 2 形式は意味上は等価です。ただし厳密には非関連の形式の方がより一般性があります、

```haskell
-- (1) 非関連の形式
type family Rep a
type instance Rep Int = Char
type instance Rep Char = Int

class Convertible a where
  convert :: a -> Rep a

instance Convertible Int where
  convert = chr

instance Convertible Char where
  convert = ord



-- (2) 関連の形式
class Convertible a where
  type Rep a
  convert :: a -> Rep a

instance Convertible Int where
  type Rep Int = Char
  convert = chr

instance Convertible Char where
  type Rep Char = Int
  convert = ord
```

多引数型クラスと関数従属の説明に使ったのと同じ例を使えば、型族の方法と関数従属性の方法との間には直接的な変換があることが分かります。これら 2 つのアプローチは同じ表現力を持つのです。

関連型族については GHCi の ``:kind!`` コマンドを使ってクエリを投げることができます。

```haskell
λ: :kind! Rep Int
Rep Int :: *
= Char
λ: :kind! Rep Char
Rep Char :: *
= Int
```

一方**データ族**を使えば、型を引数に取るデータ構成子を作ることができます。通常、型クラスの関数については、単なる型クラスの引数により定まる、一様な振る舞いをするものしか定義できません。データ族があれば、型を指標とする、型ごとに異なる振る舞いを生むことができます。

例えば、統一の API を用いながらも内部ではデータの配置が異なる、より複雑な配列の構造（ビットマスクを施した配列や、タプルの配列、……）を作りたければ、データ族を使って実現できます。

```haskell
{-# LANGUAGE TypeFamilies #-}

import qualified Data.Vector.Unboxed as V

data family Array a
data instance Array Int       = IArray (V.Vector Int)
data instance Array Bool      = BArray (V.Vector Bool)
data instance Array (a,b)     = PArray (Array a) (Array b)
data instance Array (Maybe a) = MArray (V.Vector Bool) (Array a)

class IArray a where
  index :: Array a -> Int -> a

instance IArray Int where
  index (IArray xs) i = xs V.! i

instance IArray Bool where
  index (BArray xs) i = xs V.! i

-- ペアの配列
instance (IArray a, IArray b) => IArray (a, b) where
  index (PArray xs ys) i = (index xs i, index ys i)

-- 値が欠けている配列
instance (IArray a) => IArray (Maybe a) where
  index (MArray bm xs) i =
    case bm V.! i of
      True  -> Nothing
      False -> Just $ index xs i
```

## 単射性

型族で定義される型レベルの関数は**単射**でなくても構わないので、異なる 2 つの入力型を同じ型へと対応付けることもできます。これは、単射であり（型レベルの関数でもある）型コンストラクタの振る舞いとは異なります。

例えば、``Maybe`` コンストラクタについては、``Maybe t1 ~ Maybe t2`` から ``t1 ~ t2`` が言えます。
``t1 = t2``.

```haskell
data Maybe a = Nothing | Just a
-- Maybe a ~ Maybe b ならば a ~ b

type instance F Int = Bool
type instance F Char = Bool

-- F a ~ F b であっても a ~ b とは限らない
```

## 役割

役割はデータ型の引数である型変数についてより繊細な指定を行うためのものです。

* ``nominal``
* ``representational``
* ``phantom``

これらが言語に追加されたのは、newtype と実行時表現の間の対応にまつわる、積年の割と厄介なバグを解決するためです。役割により導入された基本的な区別は、型の等価性には 2 種類の考え方があるということです。

* ``nominal``［名前上の］： 2 つの型は等しい。
* ``representational``［表現上の］： 2 つの型は同じ実行時表現を持つ。

```haskell
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

newtype Age = MkAge { unAge :: Int }

type family Inspect x
type instance Inspect Age = Int
type instance Inspect Int = Bool

class Boom a where
  boom :: a -> Inspect a

instance Boom Int where
  boom = (== 0)

deriving instance Boom Age

-- GHC 7.6.3 では未定義の振る舞いを起こす
failure = boom (MkAge 3)
-- -6341068275333450897
```

役割は通常自動的に推論されますが、``RoleAnnotations`` 拡張を使えば手動で注釈を付けることができます。例外的な場合を除けばこれは不要ですが、裏で何が起こっているのかを知るには便利です。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE RoleAnnotations #-}

data Nat = Zero | Suc Nat

type role Vec nominal representational
data Vec :: Nat -> * -> * where
  Nil  :: Vec Zero a
  (:*) :: a -> Vec n a -> Vec (Suc n) a

type role App representational nominal
data App (f :: k -> *) (a :: k) = App (f a)

type role Mu nominal nominal
data Mu (f :: (k -> *) -> k -> *) (a :: k) = Roll (f (Mu f) a)

type role Proxy phantom
data Proxy (a :: k) = Proxy
```

参照：

* [Roles: A New Feature of GHC](http://typesandkinds.wordpress.com/2013/08/15/roles-a-new-feature-of-ghc/)
* [Roles](https://ghc.haskell.org/trac/ghc/wiki/Roles)

## MonoTraversable

mono-traversableは、型族を使うことで Functor、Foldable、Traversable の概念を一般化し、単相的な型も多相的な型も扱えるようにしています。

```haskell
omap :: MonoFunctor mono => (Element mono -> Element mono) -> mono -> mono

otraverse :: (Applicative f, MonoTraversable mono)
          => (Element mono -> f (Element mono)) -> mono -> f mono

ofoldMap :: (Monoid m, MonoFoldable mono)
         => (Element mono -> m) -> mono -> m
ofoldl' :: MonoFoldable mono
        => (a -> Element mono -> a) -> a -> mono -> a
ofoldr :: MonoFoldable mono
        => (Element mono -> b -> b) -> b -> mono -> b
```

例えば、Text 型はこれまでは以上の型クラスのいずれも通常受け付けませんが、今では Foldable と Traversable のインターフェースを実現するインスタンスを書くことができます。

```haskell
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE OverloadedStrings #-}

import Data.Text
import Data.Char
import Data.Monoid
import Data.MonoTraversable
import Control.Applicative

bs :: Text
bs = "Hello Haskell."

shift :: Text
shift = omap (chr . (+1) . ord) bs
-- "Ifmmp!Ibtlfmm/"

backwards :: [Char]
backwards = ofoldl' (flip (:)) "" bs
-- ".lleksaH olleH"


data MyMonoType = MNil | MCons Int MyMonoType deriving Show

type instance Element MyMonoType = Int

instance MonoFunctor MyMonoType where
  omap f MNil = MNil
  omap f (MCons x xs) = f x `MCons` omap f xs

instance MonoFoldable MyMonoType where
  ofoldMap f = ofoldr (mappend . f) mempty
  ofoldr       = mfoldr
  ofoldl'      = mfoldl'
  ofoldr1Ex f  = ofoldr1Ex f . mtoList
  ofoldl1Ex' f = ofoldl1Ex' f . mtoList

instance MonoTraversable MyMonoType where
  omapM f xs = mapM f (mtoList xs) >>= return . mfromList
  otraverse f = ofoldr acons (pure MNil)
    where acons x ys = MCons <$> f x <*> ys

mtoList :: MyMonoType -> [Int]
mtoList (MNil) = []
mtoList (MCons x xs) = x : (mtoList xs)

mfromList :: [Int] -> MyMonoType
mfromList [] = MNil
mfromList (x:xs) = MCons x (mfromList xs)

mfoldr :: (Int -> a -> a) -> a -> MyMonoType -> a
mfoldr f z MNil =  z
mfoldr f z (MCons x xs) =  f x (mfoldr f z xs)

mfoldl' :: (a -> Int -> a) -> a -> MyMonoType -> a
mfoldl' f z MNil = z
mfoldl' f z (MCons x xs) = let z' = z `f` x
                           in seq z' $ mfoldl' f z' xs

ex1 :: Int
ex1 = mfoldl' (+) 0 (mfromList [1..25])

ex2 :: MyMonoType
ex2 = omap (+1) (mfromList [1..25])
```

参照：

* [From Semigroups to Monads](http://fundeps.com/tables/FromSemigroupToMonads.pdf)

## NonEmpty

リストが空である場合を扱うためには、Prelude に多く含まれる機能の良くない（しばしば部分的な）関数を使うよりも、空のリストが型の要素として構成されるのを静的に防いでしまうほうがよい場合もあります。

```haskell
infixr 5 :|, <|
data NonEmpty a = a :| [a]

head :: NonEmpty a -> a
toList :: NonEmpty a -> [a]
fromList :: [a] -> NonEmpty a
```

```haskell
head :: NonEmpty a -> a
head ~(a :| _) = a
```

```haskell
import Data.List.NonEmpty
import Prelude hiding (head, tail, foldl1)
import Data.Foldable (foldl1)

a :: NonEmpty Integer
a = fromList [1,2,3]
-- 1 :| [2,3]

b :: NonEmpty Integer
b = 1 :| [2,3]
-- 1 :| [2,3]

c :: NonEmpty Integer
c = fromList []
-- *** Exception: NonEmpty.fromList: empty list

d :: Integer
d = foldl1 (+) $ fromList [1..100]
-- 5050
```

GHC 7.8 では、``-XOverloadedLists``を使えば、わざわざ``fromList``や``toList``で変換しなくてもよくなります。

## 手動証明

計算機科学の最も深淵な結果の一つである、[カリー＝ハワード同型対応](https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%AA%E3%83%BC%EF%BC%9D%E3%83%8F%E3%83%AF%E3%83%BC%E3%83%89%E5%90%8C%E5%9E%8B%E5%AF%BE%E5%BF%9C) ([Curry–Howard
correspondence](https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence)) は、論理的命題は型によりモデル化できて、それらの型を実体化する過程はそれらの命題を証明している、という関係です。プログラムは証明であり、証明はプログラムなのです。

型 | 論理
--- | ---
``A`` | 命題
``a : A`` | 証明
``B(x)`` | 述語
``Void`` | ⊥（偽）
``Unit`` | ⊤（真）
``A + B`` | A ∨ B（論理和）
``A × B`` | A ∧ B（論理積）
``A -> B`` | A ⇒ B（含意）

依存型付けの言語では、これらの結果を十分に利用できます。Haskell には依存型の提供する強さが無いですが、些細な結果ならば証明できます。例えば、足し算を行う型レベルの関数をモデル化してから、ゼロが足し算の恒等元であることを示す小さい証明を書いて見せることができます。

```
P 0                   [基本の段階]
∀n. P n  → P (1+n)    [帰納的な段階]
-------------------
∀n. P(n)
```

```
公理 1: a + 0 = a
公理 2: a + suc b = suc (a + b)

  0 + suc a
= suc (0 + a)  [公理 2]
= suc a        [帰納法の仮定]
∎
```

Haskell に翻訳すると、公理は単に型の定義になり、帰納的なデータ型に対して再帰を行うことにより、証明の帰納的な段階を構成できます。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE TypeOperators #-}

data Z
data S n

data SNat n where
  Zero :: SNat Z
  Succ :: SNat n -> SNat (S n)

data Eql a b where
  Refl :: Eql a a

type family Add m n
type instance Add Z n = n
type instance Add (S m) n = S (Add m n)

add :: SNat n -> SNat m -> SNat (Add n m)
add Zero     m = m
add (Succ n) m = Succ (add n m)

cong :: Eql a b -> Eql (f a) (f b)
cong Refl = Refl

-- ∀n. 0 + suc n = suc n
plus_suc :: forall n.  SNat n
         -> Eql (Add Z (S n)) (S n)
plus_suc Zero = Refl
plus_suc (Succ n) = cong (plus_suc n)

-- ∀n. 0 + n = n
plus_zero :: forall n. SNat n
         -> Eql (Add Z n) n
plus_zero Zero = Refl
plus_zero (Succ n) = cong (plus_zero n)
```

``TypeOperators`` 拡張を使えば、型レベルで中置記法を用いることもできます。

```haskell
data a :=: b where
  Refl :: a :=: a

cong :: a :=: b -> (f a) :=: (f b)
cong Refl = Refl

type family (n :: Nat) :+ (m :: Nat) :: Nat
type instance Zero     :+ m = m
type instance (Succ n) :+ m = Succ (n :+ m)

plus_suc :: forall n m. SNat n -> SNat m -> (n :+ (S m)) :=: (S (n :+ m))
plus_suc Zero m = Refl
plus_suc (Succ n) m = cong (plus_suc n m)
```

## 制約種

また、``-XConstraintKinds``拡張を有効にすると、Haskell で量化子の範囲を限定する述語を GHC の実装で型として扱えるようになります。この拡張を使えば制約を第一級の型として使えるようになります。

```haskell
Num :: * -> Constraint
Odd :: * -> Constraint
```

```haskell
type T1 a = (Num a, Ord a)
```

空の制約の集合は``() :: Constraint``により示されます。

上手い例を出しましょう。興味のあるコンテナの要素に対する制約を持つ一般性のある ``Sized`` クラスを作りたいとすれば、型族を使えばいとも簡単に実現できます。

```haskell
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ConstraintKinds #-}

import GHC.Exts (Constraint)
import Data.Hashable
import Data.HashSet

type family Con a :: Constraint
type instance Con [a] = (Ord a, Eq a)
type instance Con (HashSet a) = (Hashable a)

class Sized a where
  gsize :: Con a => a -> Int

instance Sized [a] where
  gsize = length

instance Sized (HashSet a) where
  gsize = size
```

ユースケースの一つは、関数により型クラスの辞書を取り込み、値として具体化することです。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE KindSignatures #-}

import GHC.Exts (Constraint)

data Dict :: Constraint -> * where
  Dict :: (c) => Dict c

dShow :: Dict (Show a) -> a -> String
dShow Dict x = show x

dEqNum :: Dict (Eq a, Num a) -> a -> Bool
dEqNum Dict x = x == 0


fShow :: String
fShow = dShow Dict 10

fEqual :: Bool
fEqual = dEqNum Dict 0
```

Constraint も AnyK も Haskell の実装では種が ``BOX`` であるという点でやや独特です。

```haskell
λ: import GHC.Prim

λ: :kind AnyK
AnyK :: BOX

λ: :kind Constraint
Constraint :: BOX
```
