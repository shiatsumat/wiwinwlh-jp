# 昇格

## 高階種

Haskell の種システムは他のほとんどの言語の持たない特異な性質を持っています。Haskell では、型と型構成子を別の型へと対応付けるデータ型を構成することが出来るのです。こうしたシステムは**高階種の型**をサポートしているといいます。

Haskell のすべての型注釈は必然的に種 ``*`` を持ちますが、左辺ではいかなる項も高階種（``* -> *``）となる可能性があります。

よくある例は、``* -> *`` の種を持つモナドです。しかし、自由モナドでも高階種は見られます。

```haskell
data Free f a where
  Pure :: a -> Free f a
  Free :: f (Free f a) -> Free f a

data Cofree f a where
  Cofree :: a -> f (Cofree f a) -> Cofree f a
```

```haskell
Free :: (* -> *) -> * -> *
Cofree :: (* -> *) -> * -> *
```

例えば ``Cofree Maybe a`` は、ある単一種の型 ``a`` に対して、``Maybe :: * -> *`` で非空のリストを具現化しています。

```haskell
-- Cofree Maybe a は非空のリスト
testCofree :: Cofree Maybe Int
testCofree = (Cofree 1 (Just (Cofree 2 Nothing)))
```

## 種多相

関数を受け取って引数に適用する標準的な値レベルの関数は、普通のヒンドリー・ミルナーの方法で不変性を持たせることができます。

```haskell
app :: forall a b. (a -> b) -> a -> b
app f a = f a
```

しかし同じことを型レベルでしようとすると、適用されるコンストラクタの多相性についての情報を緩めることになります。

```haskell
-- TApp :: (* -> *) -> * -> *
data TApp f a = MkTApp (f a)
```

``-XPolyKinds`` をオンにすると、種レベルでも多相変数が使えるようになります。

```haskell
-- デフォルト：　　　(* -> *) -> * -> *
-- 多相種が有効の時：(k -> *) -> k -> *
data TApp f a = MkTApp (f a)

-- デフォルト：　　　((* -> *) -> (* -> *)) -> (* -> *)
-- 多相種が有効の時：((k -> *) -> (k -> *)) -> (k -> *)
data Mu f a = Roll (f (Mu f) a)

-- デフォルト：　　　* -> *
-- 多相種が有効の時：k -> *
data Proxy a = Proxy
```

種多相の ``Proxy``［代理］型を使えば、種として任意のアリティのコンストラクタを扱える型クラスの関数を書くことができます。

```haskell
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}

data Proxy a = Proxy
data Rep = Rep

class PolyClass a where
  foo :: Proxy a -> Rep
  foo = const Rep

-- () :: *
-- [] :: * -> *
-- Either :: * -> * -> *

instance PolyClass ()
instance PolyClass []
instance PolyClass Either
```

例えば多相の ``S`` ``K`` コンビネータを型レベルで書いて行くことができます。

```haskell
{-# LANGUAGE PolyKinds #-}

newtype I (a :: *) = I a
newtype K (a :: *) (b :: k) = K a
newtype Flip (f :: k1 -> k2 -> *) (x :: k2) (y :: k1) = Flip (f y x)

unI :: I a -> a
unI (I x) = x

unK :: K a b -> a
unK (K x) = x

unFlip :: Flip f x y -> f y x
unFlip (Flip x) = x
```

## データ種

``-XDataKinds`` 拡張を使えば、値レベルと型レベルでコンストラクタを参照することができます。例えば、単純な和型について考えてみましょう。

```haskell
data S a b = L a | R b

-- S :: * -> * -> *
-- L :: a -> S a b
-- R :: b -> S a b
```

拡張を有効にすると、型コンストラクタが昇格されて、``L`` や ``R`` は、型 ``S`` のデータコンストラクタとしても、種 ``S`` の型 ``L``/``R`` としても見ることができるようになります。

```haskell
{-# LANGUAGE DataKinds #-}

data S a b = L a | R b

-- S :: * -> * -> *
-- L :: * -> S * *
-- R :: * -> S * *
```

昇格されたデータコンストラクタは、型シグニチャでシングルクオートを前置することで参照できます。また、これも重要ですが、これらの昇格されたコンストラクタはデフォルトではモジュールではエクスポートされないけれども、型シノニムのインスタンスがこの記法を使って作成できるのです。

```haskell
data Foo = Bar | Baz
type Bar = 'Bar
type Baz = 'Baz
```

これを型族と組み合わせると、型を種レベルに持ち上げると意味のある型レベルの関数が書けるということが分かります。

```haskell
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}

import Prelude hiding (Bool(..))

data Bool = True | False

type family Not (a :: Bool) :: Bool

type instance Not True = False
type instance Not False = True

false :: Not True ~ False => a
false = undefined

true :: Not False ~ True => a
true = undefined

-- コンパイル時に失敗します。
-- Couldn't match type 'False with 'True
invalid :: Not True ~ True => a
invalid = undefined
```

## 配列

この新しい構造を使うと、要素だけでなく長さも引数に持つ ``Vec`` 型を作成することができます。一般化された代数的データ型で後者の型を種シグニチャで記述できるだけの豊かな種言語があるからです。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

data Nat = Z | S Nat deriving (Eq, Show)

type Zero  = Z
type One   = S Zero
type Two   = S One
type Three = S Two
type Four  = S Three
type Five  = S Four

data Vec :: Nat -> * -> * where
  Nil :: Vec Z a
  Cons :: a -> Vec n a -> Vec (S n) a

instance Show a => Show (Vec n a) where
  show Nil         = "Nil"
  show (Cons x xs) = "Cons " ++ show x ++ " (" ++ show xs ++ ")"

class FromList n where
  fromList :: [a] -> Vec n a

instance FromList Z where
  fromList [] = Nil

instance FromList n => FromList (S n) where
  fromList (x:xs) = Cons x $ fromList xs


lengthVec :: Vec n a -> Nat
lengthVec Nil = Z
lengthVec (Cons x xs) = S (lengthVec xs)

zipVec :: Vec n a -> Vec n b -> Vec n (a,b)
zipVec Nil Nil = Nil
zipVec (Cons x xs) (Cons y ys) = Cons (x,y) (zipVec xs ys)

vec4 :: Vec Four Int
vec4 = fromList [0, 1, 2, 3]

vec5 :: Vec Five Int
vec5 = fromList [0, 1, 2, 3, 4]


example1 :: Nat
example1 = lengthVec vec4
-- S (S (S (S Z)))

example2 :: Vec Four (Int, Int)
example2 = zipVec vec4 vec4
-- Cons (0,0) (Cons (1,1) (Cons (2,2) (Cons (3,3) (Nil))))
```

以下のように形の合わない 2 つの ``Vec`` 型をジップしようとすると、1 個ずれのエラーがコンパイル時に生じます。

```haskell
example2 = zipVec vec4 vec5
-- Couldn't match type 'S 'Z with 'Z
-- Expected type: Vec Four Int
--   Actual type: Vec Five Int
```

同様のテクニックを用いれば、空かどうかの静的なフラグを指標に持つコンテナを作成できます。これは、空のリストの頭部 (head) を得ようとするとコンパイル時のエラーが出るものです。言い換えれば、head 関数に渡す引数が空でないとを証明することをコンパイラに課しているということです。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

data Size = Empty | NonEmpty

data List a b where
  Nil  :: List Empty a
  Cons :: a -> List b a -> List NonEmpty a

head' :: List NonEmpty a -> a
head' (Cons x _) = x

example1 :: Int
example1 = head' (1 `Cons` (2 `Cons` Nil))

-- Cannot match type Empty with NonEmpty
example2 :: Int
example2 = head' Nil
```

```haskell
Couldn't match type None with Many
Expected type: List NonEmpty Int
  Actual type: List Empty Int
```

参照：

* [Giving Haskell a Promotion](https://research.microsoft.com/en-us/people/dimitris/fc-kind-poly.pdf)
* [Faking It: Simulating Dependent Types in Haskell](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.22.2636&rep=rep1&type=pdf)

## 型レベル数値

GHC の型リテラルを明示的なペアノ算術の代わりに使うこともできます。

GHC 7.6 は簡約の実行について非常に保守的ですが、GHC 7.8 はずっと改善されていて、自然数が絡む型レベルの制約の多くを解決できます。それでも少々 GHC に言い聞かせてやる必要があることがあります。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}

import GHC.TypeLits

data Vec :: Nat -> * -> * where
  Nil :: Vec 0 a
  Cons :: a -> Vec n a -> Vec (1 + n) a

-- GHC 7.6 は簡約してくれない
-- vec3 :: Vec (1 + (1 + (1 + 0))) Int

vec3 :: Vec 3 Int
vec3 = 0 `Cons` (1 `Cons` (2 `Cons` Nil))
```

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}

import GHC.TypeLits
import Data.Type.Equality

data Foo :: Nat -> * where
  Small    :: (n <= 2)  => Foo n
  Big      :: (3 <= n) => Foo n

  Empty    :: ((n == 0) ~ True) => Foo n
  NonEmpty :: ((n == 0) ~ False) => Foo n

big :: Foo 10
big = Big

small :: Foo 2
small = Small

empty :: Foo 0
empty = Empty

nonempty :: Foo 3
nonempty = NonEmpty
```

参照：

* [Type-Level Literals](http://www.haskell.org/ghc/docs/7.8.2/html/users_guide/type-level-literals.html)

## 型等価性

Haskell でもっと凝った証明を作るという話題の続きをしましょう。GHC 7.8 は最近 ``Data.Type.Equality`` モジュールを搭載しました。このモジュールは、型の相等性を値として、制約として、そして昇格したブール値として表現するための型レベルの演算を大量に提供しています。

```haskell
(~)   :: k -> k -> Constraint
(==)  :: k -> k -> Bool
(<=)  :: Nat -> Nat -> Constraint
(<=?) :: Nat -> Nat -> Bool
(+)   :: Nat -> Nat -> Nat
(-)   :: Nat -> Nat -> Nat
(*)   :: Nat -> Nat -> Nat
(^)   :: Nat -> Nat -> Nat
```

```haskell
(:~:)     :: k -> k -> *
Refl      :: a1 :~: a1
sym       :: (a :~: b) -> b :~: a
trans     :: (a :~: b) -> (b :~: c) -> a :~: c
castWith  :: (a :~: b) -> a -> b
gcastWith :: (a :~: b) -> (a ~ b => r) -> r
```

これにより、コンパイル時に検査が出来る制限を書くのにずっと強力な言語が手に入ります。この仕組みを使って、後でより高度な証明を書いていきます。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstraintKinds #-}

import GHC.TypeLits
import Data.Type.Equality

type Not a b = ((b == a) ~ False)

restrictUnit :: Not () a => a -> a
restrictUnit = id

restrictChar :: Not Char a => a -> a
restrictChar = id
```

## 代理

種多相を幽霊型と組み合わせれば、代理型 (proxy type) を表現できます。この型は引数の無いただ一つのコンストラクタを値として持ち、値が行き渡る間ずっと好きな型を保持することが出来る、種多相の幽霊型変数を持ちます。

```haskell
{-# LANGUAGE PolyKinds #-}

-- | 具体的な、種多相の代理型
data Proxy t = Proxy
```

```haskell
import Data.Proxy

a :: Proxy ()
a = Proxy

b :: Proxy 3
b = Proxy

c :: Proxy "symbol"
c = Proxy

d :: Proxy Maybe
d = Proxy

e :: Proxy (Maybe ())
e = Proxy
```

これは Prelude 7.8 で提供されています。

## 昇格された構文

コンストラクタが DataKinds により昇格されるのは見てきましたが、値レベルと同様、GHC には明示的にコンスやペアを使わなくてもリストやタプルを作れる糖衣構文があります。この構文は、``-XTypeOperators``［型演算子］拡張により有効になります。この拡張は、型レベルでリストの構文と任意のアリティのタプルを導入します。

```haskell
data HList :: [*] -> * where
  HNil  :: HList '[]
  HCons :: a -> HList t -> HList (a ': t)

data Tuple :: (*,*) -> * where
  Tuple :: a -> b -> Tuple '(a,b)
```

これを使えば、型レベルの合成されたデータを好きなように作ることができます。

```haskell
λ: :kind 1
1 :: Nat

λ: :kind "foo"
"foo" :: Symbol

λ: :kind [1,2,3]
[1,2,3] :: [Nat]

λ: :kind [Int, Bool, Char]
[Int, Bool, Char] :: [*]

λ: :kind Just [Int, Bool, Char]
Just [Int, Bool, Char] :: Maybe [*]

λ: :kind '("a", Int)
(,) Symbol *

λ: :kind [ '("a", Int), '("b", Bool) ]
[ '("a", Int), '("b", Bool) ] :: [(,) Symbol *]
```

## シングルトン型

シングルトン型は単一の値のみを持つ型です。シングルトン型を構成する方法は様々であり、GADT を使ってもデータ族を使っても可能です。

```haskell
data instance Sing (a :: Nat) where
  SZ :: Sing 'Z
  SS :: Sing n -> Sing ('S n)

data instance Sing (a :: Maybe k) where
  SNothing :: Sing 'Nothing
  SJust :: Sing x -> Sing ('Just x)

data instance Sing (a :: Bool) where
  STrue :: Sing True
  SFalse :: Sing False
```

**昇格された自然数**

値レベル | 型レベル | モデル
--- | --- | ---
``SZ`` | ``Sing 'Z`` | ``0``
``SS SZ`` | ``Sing ('S 'Z)`` | ``1``
``SS (SS SZ)`` | ``Sing ('S ('S 'Z))`` | ``2``

**昇格されたブール値**

値レベル | 型レベル | モデル
--- | --- | ---
``STrue`` | ``Sing 'False`` | ``False``
``SFalse`` | ``Sing 'True`` | ``True``

**昇格されたMaybe**

値レベル | 型レベル | モデル
--- | --- | ---
``SJust a`` | ``Sing (SJust 'a)`` | ``Just a``
``SNothing`` | ``Sing Nothing`` | ``Nothing``

シングルトン型は、Haskell で依存型らしきものを作る（すなわち値に基づく項を持つ型を構成する）というちょっとしたお手軽なお仕事で欠かせないものです。シングルトン型は、型と値の間の対応を型の構造的性質としてモデル化することで”ズルをする”ための手段なのです。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}

import Data.Proxy
import GHC.Exts (Any)
import Prelude hiding (succ)

data Nat = Z | S Nat

-- 種を指標とするデータ族
data family Sing (a :: k)

data instance Sing (a :: Nat) where
  SZ :: Sing 'Z
  SS :: Sing n -> Sing ('S n)

data instance Sing (a :: Maybe k) where
  SNothing :: Sing 'Nothing
  SJust :: Sing x -> Sing ('Just x)

data instance Sing (a :: Bool) where
  STrue :: Sing True
  SFalse :: Sing False

data Fin (n :: Nat) where
  FZ :: Fin (S n)
  FS :: Fin n -> Fin (S n)

data Vec a n where
  Nil  :: Vec a Z
  Cons :: a -> Vec a n -> Vec a (S n)

class SingI (a :: k) where
  sing :: Sing a

instance SingI Z where
  sing = SZ

instance SingI n => SingI (S n) where
  sing = SS sing

deriving instance Show Nat
deriving instance Show (SNat a)
deriving instance Show (SBool a)
deriving instance Show (Fin a)
deriving instance Show a => Show (Vec a n)

type family (m :: Nat) :+ (n :: Nat) :: Nat where
  Z :+ n = n
  S m :+ n = S (m :+ n)

type SNat (k :: Nat) = Sing k
type SBool (k :: Bool) = Sing k
type SMaybe (b :: a) (k :: Maybe a) = Sing k

size :: Vec a n -> SNat n
size Nil         = SZ
size (Cons x xs) = SS (size xs)

forget :: SNat n -> Nat
forget SZ = Z
forget (SS n) = S (forget n)

natToInt :: Integral n => Nat -> n
natToInt Z     = 0
natToInt (S n) = natToInt n + 1

intToNat :: (Integral a, Ord a) => a -> Nat
intToNat 0 = Z
intToNat n = S $ intToNat (n - 1)

sNatToInt :: Num n => SNat x -> n
sNatToInt SZ     = 0
sNatToInt (SS n) = sNatToInt n + 1

index :: Fin n -> Vec a n -> a
index FZ (Cons x _)      = x
index (FS n) (Cons _ xs) = index n xs


test1 :: Fin (S (S (S Z)))
test1 = FS (FS FZ)

test2 :: Int
test2 = index FZ (1 `Cons` (2 `Cons` Nil))

test3 :: Sing ('Just ('S ('S Z)))
test3 = SJust (SS (SS SZ))

test4 :: Sing ('S ('S Z))
test4 = SS (SS SZ)

-- 多相のコンストラクタ SingI
test5 :: Sing ('S ('S Z))
test5 = sing
```

``GHC.TypeLits`` が提供している組み込みのシングルトン型は、型レベルの値が値レベルに反映されて更にまた（存在型ではありますが）型レベルへと反映されるようにする、便利な実装を持っています。

```haskell
someNatVal :: Integer -> Maybe SomeNat
someSymbolVal :: String -> SomeSymbol

natVal :: KnownNat n => proxy n -> Integer
symbolVal :: KnownSymbol n => proxy n -> String
```

```haskell
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}

import Data.Proxy
import GHC.TypeLits

a :: Integer
a = natVal (Proxy :: Proxy 1)
-- 1

b :: String
b = symbolVal (Proxy :: Proxy "foo")
-- "foo"

c :: Integer
c = natVal (Proxy :: Proxy (2 + 3))
-- 5
```

## 閉じた型族

ここまで使ってきた型族（開いた型族と呼ばれています）には型レベルの関数で等式が使われる順番が考えられていませんでした。型族はコードのどこでも拡張できるので、解決は入手可能な定義ごとで単に順々に進みます。閉じた型族を使えば、解決のための基本ケースを作るのを許す代替の宣言が使えるようになります。この宣言を使えば、型に対する再帰関数を実際に書けるようになります。

例えば、関数の型における引数を数えて値レベルへと具体化する関数を書きたいとしましょう。

```haskell
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

import Data.Proxy
import GHC.TypeLits

type family Count (f :: *) :: Nat where
  Count (a -> b) = 1 + (Count b)
  Count x = 1

type Fn1 = Int -> Int
type Fn2 = Int -> Int -> Int -> Int

fn1 :: Integer
fn1 = natVal (Proxy :: Proxy (Count Fn1))
-- 2

fn2 :: Integer
fn2 = natVal (Proxy :: Proxy (Count Fn2))
-- 4
```

この機能によりやっと書けるようになる関数はかなり意外なものであり、型レベルでも十分な意味のある論理が書けるようになります。

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UndecidableInstances #-}

import GHC.TypeLits
import Data.Proxy
import Data.Type.Equality

-- 型レベルリストに対する型レベルの関数

type family Reverse (xs :: [k]) :: [k] where
  Reverse '[] = '[]
  Reverse xs = Rev xs '[]

type family Rev (xs :: [k]) (ys :: [k]) :: [k] where
  Rev '[] i = i
  Rev (x ': xs) i = Rev xs (x ': i)

type family Length (as :: [k]) :: Nat where
  Length '[] = 0
  Length (x ': xs) = 1 + Length xs

type family If (p :: Bool) (a :: k) (b :: k) :: k where
  If True a b = a
  If False a b = b

type family Concat (as :: [k]) (bs :: [k]) :: [k] where
  Concat a '[] = a
  Concat '[] b = b
  Concat (a ': as) bs = a ': Concat as bs

type family Map (f :: a -> b) (as :: [a]) :: [b] where
  Map f '[] = '[]
  Map f (x ': xs) = f x ': Map f xs

type family Sum (xs :: [Nat]) :: Nat where
  Sum '[] = 0
  Sum (x ': xs) = x + Sum xs

ex1 :: Reverse [1,2,3] ~ [3,2,1] => Proxy a
ex1 = Proxy

ex2 :: Length [1,2,3] ~ 3 => Proxy a
ex2 = Proxy

ex3 :: (Length [1,2,3]) ~ (Length (Reverse [1,2,3])) => Proxy a
ex3 = Proxy

-- 型レベルの計算を値レベルへと反映する
ex4 :: Integer
ex4 = natVal (Proxy :: Proxy (Length (Concat [1,2,3] [4,5,6])))
-- 6

ex5 :: Integer
ex5 = natVal (Proxy :: Proxy (Sum [1,2,3]))
-- 6

-- Couldn't match type ‘2’ with ‘1’
ex6 :: Reverse [1,2,3] ~ [3,1,2] => Proxy a
ex6 = Proxy
```

型族の関数の結果の種は ``(*)`` でなくても構いません。例えば Nat や Constraint でも良いのです。

```haskell
type family Elem (a :: k) (bs :: [k]) :: Constraint where
  Elem a (a ': bs) = (() :: Constraint)
  Elem a (b ': bs) = a `Elem` bs

type family Sum (ns :: [Nat]) :: Nat where
  Sum '[] = 0
  Sum (n ': ns) = n + Sum ns
```

## 種を指標とする型族

型クラスは通常型を指標としますが、同様に種を指標とすることもできます。種は型変数の種シグニチャで明示されることもあります。

```haskell
type family (a :: k) == (b :: k) :: Bool
type instance a == b = EqStar a b
type instance a == b = EqArrow a b
type instance a == b = EqBool a b

type family EqStar (a :: *) (b :: *) where
  EqStar a a = True
  EqStar a b = False

type family EqArrow (a :: k1 -> k2) (b :: k1 -> k2) where
  EqArrow a a = True
  EqArrow a b = False

type family EqBool a b where
  EqBool True  True  = True
  EqBool False False = True
  EqBool a     b     = False

type family EqList a b where
  EqList '[]        '[]        = True
  EqList (h1 ': t1) (h2 ': t2) = (h1 == h2) && (t1 == t2)
  EqList a          b          = False
```

## 昇格されたシンボル

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ConstraintKinds #-}


import GHC.TypeLits
import Data.Type.Equality

data Label (l :: Symbol) = Get

class Has a l b | a l -> b where
  from :: a -> Label l -> b

data Point2D = Point2 Double Double deriving Show
data Point3D = Point3 Double Double Double deriving Show

instance Has Point2D "x" Double where
  from (Point2 x _) _ = x

instance Has Point2D "y" Double where
  from (Point2 _ y) _ = y


instance Has Point3D "x" Double where
  from (Point3 x _ _) _ = x

instance Has Point3D "y" Double where
  from (Point3 _ y _) _ = y

instance Has Point3D "z" Double where
  from (Point3 _ _ z) _ = z


infixl 6 #

(#) :: a -> (a -> b) -> b
(#) = flip ($)

_x :: Has a "x" b => a -> b
_x pnt = from pnt (Get :: Label "x")

_y :: Has a "y" b => a -> b
_y pnt = from pnt (Get :: Label "y")

_z :: Has a "z" b => a -> b
_z pnt = from pnt (Get :: Label "z")

type Point a r = (Has a "x" r, Has a "y" r)

distance :: (Point a r, Point b r, Floating r) => a -> b -> r
distance p1 p2 = sqrt (d1^2 + d2^2)
  where
    d1 = (p1 # _x) + (p1 # _y)
    d2 = (p2 # _x) + (p2 # _y)

main :: IO ()
main = do
  print $ (Point2 10 20) # _x

  -- Fails with: No instance for (Has Point2D "z" a0)
  -- print $ (Point2 10 20) # _z

  print $ (Point3 10 20 30) # _x
  print $ (Point3 10 20 30) # _z

  print $ distance (Point2 1 3) (Point2 2 7)
  print $ distance (Point2 1 3) (Point3 2 7 4)
  print $ distance (Point3 1 3 5) (Point3 2 7 3)
```

レコードは基本的にはタプルと変わらないので、レコードのフィールド名についても同様の構成ができます。

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE ConstraintKinds #-}


import GHC.TypeLits

newtype Field (n :: Symbol) v = Field { unField :: v }
  deriving Show

data Person1 = Person1
  { _age      :: Field "age" Int
  , _name     :: Field "name" String
  }

data Person2 = Person2
  { _age'  :: Field "age" Int
  , _name' :: Field "name" String
  , _lib'  :: Field "lib" String
  }

deriving instance Show Person1
deriving instance Show Person2

data Label (l :: Symbol) = Get

class Has a l b | a l -> b where
  from :: a -> Label l -> b

instance Has Person1 "age" Int where
  from (Person1 a _) _ = unField a

instance Has Person1 "name" String where
  from (Person1 _ a) _ = unField a

instance Has Person2 "age" Int where
  from (Person2 a _ _) _ = unField a

instance Has Person2 "name" String where
  from (Person2 _ a _) _ = unField a

age :: Has a "age" b => a -> b
age pnt = from pnt (Get :: Label "age")

name :: Has a "name" b => a -> b
name pnt = from pnt (Get :: Label "name")

-- レコードの”サイモン性”を表す引数を持つ制約
type Simon a = (Has a "name" String, Has a "age" Int)

spj :: Person1
spj = Person1 (Field 56) (Field "Simon Peyton Jones")

smarlow :: Person2
smarlow = Person2 (Field 38) (Field "Simon Marlow") (Field "rts")


catNames :: (Simon a, Simon b) => a -> b -> String
catNames a b = name a ++ name b

addAges :: (Simon a, Simon b) => a -> b -> Int
addAges a b = age a + age b


names :: String
names = name smarlow ++ "," ++ name spj
-- "Simon Marlow,Simon Peyton Jones"

ages :: Int
ages = age spj + age smarlow
-- 94
```

特筆すべきなのは、このアプローチは大まかに言って単なるボイラープレートクラスのインスタンスを作っているに過ぎないということです。この方法は、TemplateHaskell なり Generic deriving 也を使えば抽象化できます。

## HList

不均一リスト (heterogeneous list) は、型により値の型が順序だって静的に表現されている、コンスから成るリストです。

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE KindSignatures #-}

infixr 5 :::

data HList (ts :: [ * ]) where
  Nil :: HList '[]
  (:::) :: t -> HList ts -> HList (t ': ts)

-- 最初の値が Bool 型である非空のリストの頭部を取る
headBool :: HList (Bool ': xs) -> Bool
headBool hlist = case hlist of
  (a ::: _) -> a

hlength :: HList x -> Int
hlength Nil = 0
hlength (_ ::: b) = 1 + (hlength b)


tuple :: (Bool, (String, (Double, ())))
tuple = (True, ("foo", (3.14, ())))

hlist :: HList '[Bool, String , Double , ()]
hlist = True ::: "foo" ::: 3.14 ::: () ::: Nil
```

もちろん、すぐにこういう問いが生まれます――型が不均一なのにそんなリストはどうやったら文字列として表示できるのでしょうか。この場合、型族を制約種と組み合わせて HList の各引数に Show を適用して、HList の全ての型が Show のインスタンスであるということを表す制約を生成し、Show のインスタンスを導出すればいいのです。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE UndecidableInstances #-}

import GHC.Exts (Constraint)

infixr 5 :::

data HList (ts :: [ * ]) where
  Nil :: HList '[]
  (:::) :: t -> HList ts -> HList (t ': ts)

type family Map (f :: a -> b) (xs :: [a]) :: [b]
type instance Map f '[] = '[]
type instance Map f (x ': xs) = f x ': Map f xs

type family Constraints (cs :: [Constraint]) :: Constraint
type instance Constraints '[] = ()
type instance Constraints (c ': cs) = (c, Constraints cs)

type AllHave (c :: k -> Constraint) (xs :: [k]) = Constraints (Map c xs)

showHList :: AllHave Show xs => HList xs -> [String]
showHList Nil = []
showHList (x ::: xs) = (show x) : showHList xs

instance AllHave Show xs => Show (HList xs) where
  show = show . showHList

example1 :: HList '[Bool, String , Double , ()]
example1 = True ::: "foo" ::: 3.14 ::: () ::: Nil
-- ["True","\"foo\"","3.14","()"]
```

## 型レベルマップ

昇格の議論のほとんどにおいて、型レベルでコンパイル時も情報を保持するデータ構造を作れるか、という問題を避けてきました。例えば型レベルの関連リストにより、型レベルのシンボルと任意の昇格できる値とを対応付けるマップを具現化できます。型族と合わせれば、型レベルの走査と検索関数を書けます。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE UndecidableInstances #-}

import GHC.TypeLits
import Data.Proxy
import Data.Type.Equality

type family If (p :: Bool) (a :: k) (b :: k) :: k where
  If True a b = a
  If False a b = b

type family Lookup (k :: a) (ls :: [(a, b)]) :: Maybe b where
  Lookup k '[] = 'Nothing
  Lookup k ('(a, b) ': xs) = If (a == k) ('Just b) (Lookup k xs)

type M = [
    '("a", 1)
  , '("b", 2)
  , '("c", 3)
  , '("d", 4)
  ]

type K = "a"
type (!!) m (k :: Symbol) a = (Lookup k m) ~ Just a

value :: Integer
value = natVal ( Proxy :: (M !! "a") a => Proxy a )
```

もし GHC に型シグニチャを展開するように頼めば、型レベルのマップの検索関数の詳細な実装を見ることができます。

```haskell
(!!)
  :: If
       (GHC.TypeLits.EqSymbol "a" k)
       ('Just 1)
       (If
          (GHC.TypeLits.EqSymbol "b" k)
          ('Just 2)
          (If
             (GHC.TypeLits.EqSymbol "c" k)
             ('Just 3)
             (If (GHC.TypeLits.EqSymbol "d" k) ('Just 4) 'Nothing)))
     ~ 'Just v =>
     Proxy k -> Proxy v
```

## 高度な証明

長さを指標に持つ配列が作れましたから、反転する関数を書いてみましょう。どれくらい難しいのでしょうか？

では、こんな感じのものを書いてみましょう。

```haskell
reverseNaive :: forall n a. Vec a n -> Vec a n
reverseNaive xs = go Nil xs -- Error: n + 0 != n
  where
    go :: Vec a m -> Vec a n -> Vec a (n :+ m)
    go acc Nil = acc
    go acc (Cons x xs) = go (Cons x acc) xs -- Error: n + succ m != succ (n + m)
```

実行すると、GHC はコードのうち 2 行に不満があると言ってきます。

```haskell
Couldn't match type ‘n’ with ‘n :+ 'Z’
    Expected type: Vec a n
      Actual type: Vec a (n :+ 'Z)

Could not deduce ((n1 :+ 'S m) ~ 'S (n1 :+ m))
    Expected type: Vec a1 (k :+ m)
      Actual type: Vec a1 (n1 :+ 'S m)
```

配列から要素を引き出していくと、配列の各要素を逆向きに組み合わせていく間に、インデクスに対して山ほどの型レベル算術をする羽目になります。しかし結果として GHC はいくつかの単一化エラーを起こすことになります。なぜなら自然数の基本的な算術の性質を GHC が知らないからです。その性質というのは ``forall n. n + 0 = 0`` と ``forall n m. n + (1 + m) = 1 + (n + m)`` のことです。もちろん、型レベルで直感的に算術をモデル化しているシステムを構成したと本気で仮定してはなりませんが、GHC はただの馬鹿なコンパイラに過ぎないので、自動的に自然数とペアノ数の間の同型を導き出すことはできないのです。

ですから、それぞれの呼び出し場所で、証明項を構成するという証明義務が生じます。この証明項は、問題となっている項の型シグニチャを組み直して、GHC が出すエラーメッセージにおける本当の型 (actual type) が期待される型 (expected type) と一致して、プログラムが完成するようにするものでなくてはなりません。

GADT で命題的相等性を導くという先ほどの議論を思い返してください。実際に証明のための機構は手に入っているのです！

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ExplicitForAll #-}

import Data.Type.Equality

data Nat = Z | S Nat

data SNat n where
  Zero :: SNat Z
  Succ :: SNat n -> SNat (S n)

data Vec :: * -> Nat -> * where
  Nil :: Vec a Z
  Cons :: a -> Vec a n -> Vec a (S n)

instance Show a => Show (Vec a n) where
  show Nil         = "Nil"
  show (Cons x xs) = "Cons " ++ show x ++ " (" ++ show xs ++ ")"

type family (m :: Nat) :+ (n :: Nat) :: Nat where
  Z :+ n = n
  S m :+ n = S (m :+ n)

-- (a ~ b) implies (f a ~ f b)
cong :: a :~: b -> f a :~: f b
cong Refl = Refl

-- (a ~ b) implies (f a) implies (f b)
subst :: a :~: b -> f a -> f b
subst Refl = id

plus_zero :: forall n. SNat n -> (n :+ Z) :~: n
plus_zero Zero = Refl
plus_zero (Succ n) = cong (plus_zero n)

plus_suc :: forall n m. SNat n -> SNat m -> (n :+ (S m)) :~: (S (n :+ m))
plus_suc Zero m = Refl
plus_suc (Succ n) m = cong (plus_suc n m)

size :: Vec a n -> SNat n
size Nil         = Zero
size (Cons _ xs) = Succ $ size xs

reverse :: forall n a. Vec a n -> Vec a n
reverse xs = subst (plus_zero (size xs)) $ go Nil xs
  where
    go :: Vec a m -> Vec a k -> Vec a (k :+ m)
    go acc Nil = acc
    go acc (Cons x xs) = subst (plus_suc (size xs) (size acc)) $ go (Cons x acc) xs

append :: Vec a n -> Vec a m -> Vec a (n :+ m)
append (Cons x xs) ys = Cons x (append xs ys)
append Nil         ys = ys

vec :: Vec Int (S (S (S Z)))
vec = 1 `Cons` (2 `Cons` (3 `Cons` Nil))

test :: Vec Int (S (S (S Z)))
test = Main.reverse vec
```

シングルトンなどという仕掛けを使わずに、型レベル自然数だけを使って何とかならないだろうか、と考える人もいるでしょう。技術的にはそうしたことは簡単であるはずです。しかしどうやら、GHC 7.8 の自然数ソルバはある性質は決定できますが、反転関数で自然数に関する証明を完成するための性質は決定できません。

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

import Prelude hiding (Eq)
import GHC.TypeLits
import Data.Type.Equality

type Z = 0

type family S (n :: Nat) :: Nat where
  S n = n + 1

-- OK!
eq_zero :: Z :~: Z
eq_zero = Refl

-- OK!
zero_plus_one :: (Z + 1) :~: (1 + Z)
zero_plus_one = Refl

-- OK!
plus_zero :: forall n. (n + Z) :~: n
plus_zero = Refl

-- OK!
plus_one :: forall n. (n + S Z) :~: S n
plus_one = Refl

-- ダメ。
plus_suc :: forall n m. (n + (S m)) :~: (S (n + m))
plus_suc = Refl
```

ただし、GHC 7.6 でこうしたことをする方法があるかもしれません。GHC 7.10 ではこれらの問題を解決することが出来るであろう、ソルバに対する変更がいくつか計画に上がっています。特に、付け外しが出来る型システムの拡張を使えるようにするという計画があります。この拡張は、こうした問題をサードパーティーの SMT ソルバに委託して、こうした数値に関する関係を導き出してもらって、GHC の型検査機にその情報を戻すというものです。

余談ですが、Agda で等価な証明を直接書き直したものがあります。これらは同様の方法で実現されていますが、依存型が無いせいで並べていた御託は一切無くなります。

```agda
module Vector where

infixr 10 _∷_

data ℕ : Set where
  zero : ℕ
  suc  : ℕ → ℕ

{-# BUILTIN NATURAL ℕ    #-}
{-# BUILTIN ZERO    zero #-}
{-# BUILTIN SUC     suc  #-}

infixl 6 _+_

_+_ : ℕ → ℕ → ℕ
0 + n = n
suc m + n = suc (m + n)

data Vec (A : Set) : ℕ → Set where
  []  : Vec A 0
  _∷_ : ∀ {n} → A → Vec A n → Vec A (suc n)

_++_ : ∀ {A n m} → Vec A n → Vec A m → Vec A (n + m)
[] ++ ys = ys
(x ∷ xs) ++ ys = x ∷ (xs ++ ys)

infix 4 _≡_

data _≡_ {A : Set} (x : A) : A → Set where
  refl : x ≡ x

subst : {A : Set} → (P : A → Set) → ∀{x y} → x ≡ y → P x → P y
subst P refl p = p

cong : {A B : Set} (f : A → B) → {x y : A} → x ≡ y → f x ≡ f y
cong f refl = refl

vec : ∀ {A} (k : ℕ) → Set
vec {A} k = Vec A k

plus_zero : {n : ℕ} → n + 0 ≡ n 
plus_zero {zero}  = refl
plus_zero {suc n} = cong suc plus_zero

plus_suc : {n : ℕ} → n + (suc 0) ≡ suc n 
plus_suc {zero}  = refl
plus_suc {suc n} = cong suc (plus_suc {n})

reverse : ∀ {A n} → Vec A n → Vec A n
reverse []       = []
reverse {A} {suc n} (x ∷ xs) = subst vec (plus_suc {n}) (reverse xs ++ (x  ∷ []))
```

## 高階種

## 種多相

## データ種

## 配列

## 型レベル数値

## 型等価性

## 代理

## 昇格された構文

## シングルトン型

## 閉じた型族

## 種を指標とする型族

## 昇格されたシンボル

## HList

## 型レベルマップ

## 高度な証明
