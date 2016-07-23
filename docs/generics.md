# ジェネリクス

## はじめに

Haskell には、様々な仕事について、そのための型クラスを自動生成するテクニックがいくつかあります。その仕事の大部分は、定型的なコードの生成です。例えば以下のようなものがあります。

* Pretty Printing
* Equality
* Serialization
* Ordering
* Traversal

## Typeable

``Typeable`` クラスを使えば、任意の型に対して実行時の情報を作れます。

```haskell
typeOf :: Typeable a => a -> TypeRep
```

```haskell
{-# LANGUAGE DeriveDataTypeable #-}

import Data.Typeable

data Animal = Cat | Dog deriving Typeable
data Zoo a = Zoo [a] deriving Typeable

equal :: (Typeable a, Typeable b) => a -> b -> Bool
equal a b = typeOf a == typeOf b

example1 :: TypeRep
example1 = typeOf Cat
-- Animal

example2 :: TypeRep
example2 = typeOf (Zoo [Cat, Dog])
-- Zoo Animal

example3 :: TypeRep
example3 = typeOf ((1, 6.636e-34, "foo") :: (Int, Double, String))
-- (Int,Double,[Char])

example4 :: Bool
example4 = equal False ()
-- False
```

Typeable のインスタンスを使えば、``unsafeCoerce`` を安全に使って、型安全なキャスト関数を書くことができます。結果の型が入力と一致しているということの証明も得られます。

```haskell
cast :: (Typeable a, Typeable b) => a -> Maybe b
cast x
  | typeOf x == typeOf ret = Just ret
  | otherwise = Nothing
  where
    ret = unsafeCoerce x
```

歴史的なことを述べておきましょう。GHC 7.6 では自分で Typeable クラスを書くことが可能ですが、GHC 以外が書くのは良くありません。しかし、GHC 7.8 では Typeable のインスタンスを手で書くのは禁止されています。

参照：

* [Typeable and Data in Haskell](http://chrisdone.com/posts/data-typeable)

## Dynamic

実行時の型情報を尋ねる方法があるので、この仕掛けを使って ``Dynamic``［動的］型を実装できます。任意の単相型を包んで一種類の型へと変換できるようにするのです。この型は、Dynamic 型を受け取るあらゆる関数に渡せます。Dynamic 型を受け取る関数は型安全な方法で内側にある値をアンパックできます。

```haskell
toDyn :: Typeable a => a -> Dynamic
fromDyn :: Typeable a => Dynamic -> a -> a
fromDynamic :: Typeable a => Dynamic -> Maybe a
cast :: (Typeable a, Typeable b) => a -> Maybe b
```

```haskell
import Data.Dynamic
import Data.Maybe

dynamicBox :: Dynamic
dynamicBox = toDyn (6.62 :: Double)

example1 :: Maybe Int
example1 = fromDynamic dynamicBox
-- Nothing

example2 :: Maybe Double
example2 = fromDynamic dynamicBox
-- Just 6.62

example3 :: Int
example3 = fromDyn dynamicBox 0
-- 0

example4 :: Double
example4 = fromDyn dynamicBox 0.0
-- 6.62
```

GHC 7.8 では Typeable クラスは種多相なので、多相関数を動的なオブジェクトに適用することができます。

## Data

Typeableが必要な時に実行時型情報を作成してくれたのと同様、Dataクラスを使えば必要な時に実行時にデータ型の構造についての情報を把握できるようになります。

```haskell
class Typeable a => Data a where
  gfoldl  :: (forall d b. Data d => c (d -> b) -> d -> c b)
          -> (forall g. g -> c g)
          -> a
          -> c a

  gunfold :: (forall b r. Data b => c (b -> r) -> c r)
          -> (forall r. r -> c r)
          -> Constr
          -> c a

  toConstr :: a -> Constr
  dataTypeOf :: a -> DataType
  gmapQl :: (r -> r' -> r) -> r -> (forall d. Data d => d -> r') -> a -> r
```

``gfoldl`` や ``gunfold`` に付いている型はちょっと恐ろしいです（しかも ``Rank2Types`` を使っています）が、これを理解する最善の方法はいくつか例を見てみることです。まずは最も些細なものから。単純な``Animal`` は以下のコードを作ります。

```haskell
data Animal = Cat | Dog deriving Typeable
```

```haskell
instance Data Animal where
  gfoldl k z Cat = z Cat
  gfoldl k z Dog = z Dog

  gunfold k z c
    = case constrIndex c of
        1 -> z Cat
        2 -> z Dog

  toConstr Cat = cCat
  toConstr Dog = cDog

  dataTypeOf _ = tAnimal

tAnimal :: DataType
tAnimal = mkDataType "Main.Animal" [cCat, cDog]

cCat :: Constr
cCat = mkConstr tAnimal "Cat" [] Prefix

cDog :: Constr
cDog = mkConstr tAnimal "Dog" [] Prefix
```

非空のコンテナ型については、もう少し面白い情報が得られます。リスト型について考えてみましょう。

```haskell
instance Data a => Data [a] where
  gfoldl _ z []     = z []
  gfoldl k z (x:xs) = z (:) `k` x `k` xs

  toConstr []    = nilConstr
  toConstr (_:_) = consConstr

  gunfold k z c
    = case constrIndex c of
        1 -> z []
        2 -> k (k (z (:)))

  dataTypeOf _ = listDataType

nilConstr :: Constr
nilConstr = mkConstr listDataType "[]" [] Prefix

consConstr :: Constr
consConstr = mkConstr listDataType "(:)" [] Infix

listDataType :: DataType
listDataType = mkDataType "Prelude.[]" [nilConstr,consConstr]
```

``gfoldl``を見れば、Data にはコンストラクタの要素でアプリカティブな道筋を辿るための関数が実装されているのだと分かるでしょう。この関数は、各要素に ``k`` を適用し、要の部分だけ ``z`` を適用しているのです。例として、2 つ組に対するインスタンスも見てみましょう。

```haskell
instance (Data a, Data b) => Data (a,b) where
  gfoldl k z (a,b) = z (,) `k` a `k` b

  toConstr (_,_) = tuple2Constr

  gunfold k z c
    = case constrIndex c of
      1 -> k (k (z (,)))

  dataTypeOf _  = tuple2DataType

tuple2Constr :: Constr
tuple2Constr = mkConstr tuple2DataType "(,)" [] Infix

tuple2DataType :: DataType
tuple2DataType = mkDataType "Prelude.(,)" [tuple2Constr]
```

これはかなりきっちりしています。たった一つの型クラスにおいて、任意の ``Data`` インスタンスの中を見て、部分項の構造や型に依存する論理を書くことができる、一般的な方法を手に入れました。任意の Data インスタンスを走査して実行時の型でパターンマッチして値をいじることができる関数が書けるようになったのです。n 個組でもリストでも ``Val`` 型の値を増やせる関数 ``over`` を書いていきましょう！

```haskell
{-# LANGUAGE DeriveDataTypeable #-}

import Data.Data
import Control.Monad.Identity
import Control.Applicative

data Animal = Cat | Dog deriving (Data, Typeable)

newtype Val = Val Int deriving (Show, Data, Typeable)

incr :: Typeable a => a -> a
incr = maybe id id (cast f)
  where f (Val x) = Val (x * 100)

over :: Data a => a -> a
over x = runIdentity $ gfoldl cont base (incr x)
  where
    cont k d = k <*> (pure $ over d)
    base = pure


example1 :: Constr
example1 = toConstr Dog
-- Dog

example2 :: DataType
example2 = dataTypeOf Cat
-- DataType {tycon = "Main.Animal", datarep = AlgRep [Cat,Dog]}

example3 :: [Val]
example3 = over [Val 1, Val 2, Val 3]
-- [Val 100,Val 200,Val 300]

example4 :: (Val, Val, Val)
example4 = over (Val 1, Val 2, Val 3)
-- (Val 100,Val 200,Val 300)
```

データ型の引数の個数を数えるジェネリックな演算を書くこともできます。

```haskell
numHoles :: Data a => a -> Int
numHoles = gmapQl (+) 0 (const 1)

example1 :: Int
example1 = numHoles (1,2,3,4,5,6,7)
-- 7

example2 :: Int
example2 = numHoles (Just 3)
-- 1
```

この方法はジェネリックな操作にも使えますが、畳み込みや危険な型強制が絡むもっと複雑なことをしようとすると、一瞬のうちに型がかなり厄介なものになってしまいます。

## Generic

ジェネリックプログラミングをするための方法で最も現代的では、型族を使うことで任意の型クラスの構造的な性質を導出するより良い方法を手に入れます。Generic は関連する型 ``Rep`` (Representation［表象］) に加えて、その関連する型と導出される型とを相互変換する、互いに逆方向で逆関数になっている関数の組（同型）を提供しています。

```haskell
class Generic a where
  type Rep a
  from :: a -> Rep a
  to :: Rep a -> a

class Datatype d where
  datatypeName :: t d f a -> String
  moduleName :: t d f a -> String

class Constructor c where
  conName :: t c f a -> String
```

[GHC.Generics](https://www.haskell.org/ghc/docs/7.4.1/html/libraries/ghc-prim-0.2.0.0/GHC-Generics.html) には、Haskell で得られる型についての様々な構造的性質をモデル化するための名前の付いた型がいくつか定義されています。

```haskell
-- | 和：コンストラクタ間の選択を表す
infixr 5 :+:
data (:+:) f g p = L1 (f p) | R1 (g p)

-- | 積：コンストラクタに複数の引数があることを表す
infixr 6 :*:
data (:*:) f g p = f p :*: g p

-- | M1 のタグ：データ型
data D
-- | M1 のタグ：コンストラクタ
data C

-- | 種 * の定数、追加の引数、そして再帰
newtype K1 i c p = K1 { unK1 :: c }

-- | メタ情報（コンストラクタの名前など）
newtype M1 i c f p = M1 { unM1 :: f p }

-- | データ型のメタ情報を表す型シノニム
type D1 = M1 D

-- | コンストラクタのメタ情報を表す型シノニム
type C1 = M1 C
```

導出の仕組みを使えば GHC は機械的に Generic のインスタンスを生成できますが、仮に単純な型に対して手でインスタンスを書こうとすればこんな感じでしょう。

```haskell
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}

import GHC.Generics

data Animal
  = Dog
  | Cat

instance Generic Animal where
  type Rep Animal = D1 T_Animal ((C1 C_Dog U1) :+: (C1 C_Cat U1))

  from Dog = M1 (L1 (M1 U1))
  from Cat = M1 (R1 (M1 U1))

  to (M1 (L1 (M1 U1))) = Dog
  to (M1 (R1 (M1 U1))) = Cat

data T_Animal
data C_Dog
data C_Cat

instance Datatype T_Animal where
  datatypeName _ = "Animal"
  moduleName _ = "Main"

instance Constructor C_Dog where
  conName _ = "Dog"

instance Constructor C_Cat where
  conName _ = "Cat"
```

``kind!``をGHCiで使えば、Genericのインスタンスに関連する型族 ``Rep`` を見ることができます。

```haskell
λ: :kind! Rep Animal
Rep Animal :: * -> *
= M1 D T_Animal (M1 C C_Dog U1 :+: M1 C C_Cat U1)

λ: :kind! Rep ()
Rep () :: * -> *
= M1 D GHC.Generics.D1() (M1 C GHC.Generics.C1_0() U1)

λ: :kind! Rep [()]
Rep [()] :: * -> *
= M1
    D
    GHC.Generics.D1[]
    (M1 C GHC.Generics.C1_0[] U1
     :+: M1
           C
           GHC.Generics.C1_1[]
           (M1 S NoSelector (K1 R ()) :*: M1 S NoSelector (K1 R [()])))
```

これで頭の良いちょっとした技が使えるようになります。データ型に対してジェネリックな関数を書く代わりに、Rep に対してそうした関数を書いてから ``from`` を使って具体化すればいいのです。Haskell のデフォルトの ``Eq`` と等価なものを、代わりにジェネリックな導出を使って書きたければ、こうすればいいのです。

```haskell
class GEq' f where
  geq' :: f a -> f a -> Bool

instance GEq' U1 where
  geq' _ _ = True

instance (GEq c) => GEq' (K1 i c) where
  geq' (K1 a) (K1 b) = geq a b

instance (GEq' a) => GEq' (M1 i c a) where
  geq' (M1 a) (M1 b) = geq' a b

-- 和に対する相等性
instance (GEq' a, GEq' b) => GEq' (a :+: b) where
  geq' (L1 a) (L1 b) = geq' a b
  geq' (R1 a) (R1 b) = geq' a b
  geq' _      _      = False

-- 積に対する相等性
instance (GEq' a, GEq' b) => GEq' (a :*: b) where
  geq' (a1 :*: b1) (a2 :*: b2) = geq' a1 a2 && geq' b1 b2
```

クラスを書くための 2 つの方法（ジェネリックな導出とカスタムの実装）の両方を使えるようにしたければ、``DefaultSignatures`` 拡張を使って、ユーザーが型クラスの関数を空白のままにして Generic に従うことも自分で定義することも可能なようにすることができます。

```haskell
{-# LANGUAGE DefaultSignatures #-}

class GEq a where
  geq :: a -> a -> Bool

  default geq :: (Generic a, GEq' (Rep a)) => a -> a -> Bool
  geq x y = geq' (from x) (from y)
```

このライブラリを使う人は、``GEq`` に対してボイラープレートを書かなくても、Generic を導出して、型クラスのインスタンスに空のインスタンスを作りさえすればいいのです。

参照：

* [Andres Loh: Datatype-generic Programming in Haskell](http://www.andres-loeh.de/DGP-Intro.pdf)
* [generic-deriving](http://hackage.haskell.org/package/generic-deriving-1.6.3)

## ジェネリックな導出

ジェネクリクスを使えば、GHCに多くの非自明なコードを生成させることができます。これは実用上見事に役立ちます。いくつかの現実世界の例を紹介しましょう。

[hashable](http://hackage.haskell.org/package/hashable)を使えば、ハッシュ関数を導出することができます。

```haskell
{-# LANGUAGE DeriveGeneric #-}

import GHC.Generics (Generic)
import Data.Hashable

data Color = Red | Green | Blue deriving (Generic, Show)

instance Hashable Color where

example1 :: Int
example1 = hash Red
-- 839657738087498284

example2 :: Int
example2 = hashWithSalt 0xDEADBEEF Red
-- 62679985974121021
```

[cereal](http://hackage.haskell.org/package/cereal)ライブラリを使えば、バイナリの表現を自動で導出できます。

```haskell
{-# LANGUAGE DeriveGeneric #-}

import Data.Word
import Data.ByteString
import Data.Serialize

import GHC.Generics

data Val = A [Val] | B [(Val, Val)] | C
  deriving (Generic, Show)

instance Serialize Val where

encoded :: ByteString
encoded = encode (A [B [(C, C)]])
-- "\NUL\NUL\NUL\NUL\NUL\NUL\NUL\NUL\SOH\SOH\NUL\NUL\NUL\NUL\NUL\NUL\NUL\SOH\STX\STX"

bytes :: [Word8]
bytes = unpack encoded
-- [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,2,2]

decoded :: Either String Val
decoded = decode encoded
```

[aeson](http://hackage.haskell.org/package/aeson) ライブラリを使えば、JSON のインスタンスに対して JSON の表現を導出できます。

```haskell
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Data.Aeson
import GHC.Generics

data Point = Point { _x :: Double, _y :: Double }
   deriving (Show, Generic)

instance FromJSON Point
instance ToJSON Point

example1 :: Maybe Point
example1 = decode "{\"x\":3.0,\"y\":-1.0}"

example2 = encode $ Point 123.4 20
```

参照：

* [A Generic Deriving Mechanism for Haskell](http://dreixel.net/research/pdf/gdmh.pdf)

## Uniplate

uniplateは任意のデータ構造に対して走査と変換の関数を記述する、ジェネリクスのライブラリです。AST の変換を書いてシステムを書き換えるのに非常に有用です。

```haskell
plate :: from -> Type from to
(|*)  :: Type (to -> from) to -> to -> Type from to
(|-)  :: Type (item -> from) to -> item -> Type from to

descend   :: Uniplate on => (on -> on) -> on -> on
transform :: Uniplate on => (on -> on) -> on -> on
rewrite   :: Uniplate on => (on -> Maybe on) -> on -> on
```

``descend`` 関数は式の直下の各子孫に関数を適用して親の式に結果を集めてきます。

``transform`` 関数は式の全ての項をボトムアップで変換する一つの道筋を進みます。

``rewrite`` 関数は式の全ての項を不動点まで完全に変換し尽くします。Maybe は停止を表しています。

```haskell
import Data.Generics.Uniplate.Direct

data Expr a
  = Fls
  | Tru
  | Var a
  | Not (Expr a)
  | And (Expr a) (Expr a)
  | Or  (Expr a) (Expr a)
  deriving (Show, Eq)

instance Uniplate (Expr a) where
  uniplate (Not f)     = plate Not |* f
  uniplate (And f1 f2) = plate And |* f1 |* f2
  uniplate (Or f1 f2)  = plate Or |* f1 |* f2
  uniplate x           = plate x

simplify :: Expr a -> Expr a
simplify = transform simp
 where
   simp (Not (Not f)) = f
   simp (Not Fls) = Tru
   simp (Not Tru) = Fls
   simp x = x

reduce :: Show a => Expr a -> Expr a
reduce = rewrite cnf
  where
    -- 二重否定
    cnf (Not (Not p)) = Just p

    -- ドモルガン
    cnf (Not (p `Or` q))  = Just $ (Not p) `And` (Not q)
    cnf (Not (p `And` q)) = Just $ (Not p) `Or` (Not q)

    -- 論理積の分配則
    cnf (p `Or` (q `And` r)) = Just $ (p `Or` q) `And` (p `Or` r)
    cnf ((p `And` q) `Or` r) = Just $ (p `Or` q) `And` (p `Or` r)
    cnf _ = Nothing


example1 :: Expr String
example1 = simplify (Not (Not (Not (Not (Var "a")))))
-- Var "a"

example2 :: [String]
example2 = [a | Var a <- universe ex]
  where
    ex = Or (And (Var "a") (Var "b")) (Not (And (Var "c") (Var "d")))
-- ["a","b","c","d"]

example3 :: Expr String
example3 = reduce $ ((a `And` b) `Or` (c `And` d)) `Or` e
  where
    a = Var "a"
    b = Var "b"
    c = Var "c"
    d = Var "d"
    e = Var "e"
```

別の方法として、Uniplate のインスタンスを Data のインスタンスから自動的に、Uniplate のインスタンスを明示的に書く必要なしに導出することもできます。このアプローチでは、明示的な手書きのインスタンスより少しだけオーバーヘッドがかさみます。

```haskell
import Data.Data
import Data.Typeable
import Data.Generics.Uniplate.Data

data Expr a
  = Fls
  | Tru
  | Lit a
  | Not (Expr a)
  | And (Expr a) (Expr a)
  | Or (Expr a) (Expr a)
  deriving (Data, Typeable, Show, Eq)
```

### Biplate

Biplateでは、ターゲットの型が元の型と同じである必要のないプレートを一般化していて、サブターゲットの型を示すために多引数型クラスを使っています。Uniplate の関数はすべて Biplate においても、対応する一般化された形の関数を持ちます。

```haskell
descendBi   :: Biplate from to => (to -> to) -> from -> from
transformBi :: Biplate from to => (to -> to) -> from -> from
rewriteBi   :: Biplate from to => (to -> Maybe to) -> from -> from

descendBiM   :: (Monad m, Biplate from to) => (to -> m to) -> from -> m from
transformBiM :: (Monad m, Biplate from to) => (to -> m to) -> from -> m from
rewriteBiM   :: (Monad m, Biplate from to) => (to -> m (Maybe to)) -> from -> m from
```

```haskell
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}

import Data.Generics.Uniplate.Direct

type Name = String

data Expr
  = Var Name
  | Lam Name Expr
  | App Expr Expr
  deriving Show

data Stmt
  = Decl [Stmt]
  | Let Name Expr
  deriving Show

instance Uniplate Expr where
  uniplate (Var x  ) = plate Var |- x
  uniplate (App x y) = plate App |* x |* y
  uniplate (Lam x y) = plate Lam |- x |* y

instance Biplate Expr Expr where
  biplate = plateSelf

instance Uniplate Stmt where
  uniplate (Decl x  ) = plate Decl ||* x
  uniplate (Let x y) = plate Let |-  x |- y

instance Biplate Stmt Stmt where
  biplate = plateSelf

instance Biplate Stmt Expr where
  biplate (Decl x) = plate Decl ||+ x
  biplate (Let x y) = plate Let |- x |* y

rename :: Name -> Name -> Expr -> Expr
rename from to = rewrite f
  where
    f (Var a) | a == from = Just (Var to)
    f (Lam a b) | a == from = Just (Lam to b)
    f _ = Nothing

s, k, sk :: Expr
s = Lam "x" (Lam "y" (Lam "z" (App (App (Var "x") (Var "z")) (App (Var "y") (Var "z")))))
k = Lam "x" (Lam "y" (Var "x"))
sk = App s k

m :: Stmt
m = descendBi f $ Decl [ (Let "s" s) , Let "k" k , Let "sk" sk ]
  where
    f = rename "x" "a"
      . rename "y" "b"
      . rename "z" "c"
```

## はじめに

## Typeable

## Dynamic

## Data

## Generic

## ジェネリックな導出

## uniplate
