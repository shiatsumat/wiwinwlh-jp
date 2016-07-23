# GADTs

## 基本

**一般化された代数的データ型** (Generalized Algebraic Data type, GADT) は、代数的データ型を拡張し、データ型のコンストラクタに型の相等性の制約を付加できるようにして、ありふれた代数的データ型では表現できない型を作れるようにしたものです。

``-XGADTs`` はデータ型の宣言に対する代替構文（``-XGADTSyntax``）を暗黙のうちに有効にしています。以下の 2 つの宣言が等価になるのです。

```haskell
-- 普通の構文
data List a
  = Empty
  | Cons a (List a)

-- GADT の構文
data List a where
  Empty :: List a
  Cons :: a -> List a -> List a
```

例えば、データ型 ``Term`` を考えると、あらゆる型になりうる ``a`` でパラメータ化された ``Term`` を受け取り、それを ``Succ`` するような項があることになります。評価器を書こうとすると、``a ~ Bool`` と ``a ~ Int`` が衝突し、問題が生じます。

```haskell
data Term a
  = Lit a
  | Succ (Term a)
  | IsZero (Term a)

-- 上手く型付けできません (>_<)
eval (Lit i)      = i
eval (Succ t)     = 1 + eval t
eval (IsZero i)   = eval i == 0
```

意味の無い項を作ることも許してしまい、エラー処理のための場合分けを増やすことになります。

```haskell
-- 有効な型
failure = Succ ( Lit True )
```

GADT を使えば、私たちの言語（つまり型安全な式だけが表現可能な言語）で型の不変条件を表現することができます。それゆえ GADT に対するパターンマッチでは、明示的にタグを付ける必要なしに型相等性の制約を付けることができます。

```haskell
{-# Language GADTs #-}

data Term a where
  Lit    :: a -> Term a
  Succ   :: Term Int -> Term Int
  IsZero :: Term Int -> Term Bool
  If     :: Term Bool -> Term a -> Term a -> Term a

eval :: Term a -> a
eval (Lit i)      = i                                   -- Term a
eval (Succ t)     = 1 + eval t                          -- Term (a ~ Int)
eval (IsZero i)   = eval i == 0                         -- Term (a ~ Int)
eval (If b e1 e2) = if eval b then eval e1 else eval e2 -- Term (a ~ Bool)

example :: Int
example = eval (Succ (Succ (Lit 3)))
```

今度はこうなります：

```haskell
-- これはコンパイル時に拒否されます
failure = Succ ( Lit True )
```

明示的な相等性制約（``a ~ b``）を関数の環境 (context) に追加することができます。例えば、以下のものは同じ型へと展開されます。

```haskell
f :: a -> a -> (a, a)
f :: (a ~ b) => a -> b -> (a,b)
```

```haskell
(Int ~ Int)  => ...
(a ~ Int)    => ...
(Int ~ a)    => ...
(a ~ b)      => ...
(Int ~ Bool) => ... -- 型検査が通らない
```

これ（暗に相等性の項を受け渡しして引き回していくこと）は実際に、GHC が GADT を実装するために裏で行っている、実装上の詳細です。しようとすれば、相等性制約と存在量化だけを使って GHC がしているのと同じことをすることもできます。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ExistentialQuantification #-}

-- 制約を使うと
data Exp a
  = (a ~ Int) => LitInt a
  | (a ~ Bool) => LitBool a
  | forall b. (b ~ Bool) => If (Exp b) (Exp a) (Exp a)

-- GADT を使うと
-- data Exp a where
--   LitInt  :: Int  -> Exp Int
--   LitBool :: Bool -> Exp Bool
--   If      :: Exp Bool -> Exp a -> Exp a -> Exp a

eval :: Exp a -> a
eval e = case e of
  LitInt i   -> i
  LitBool b  -> b
  If b tr fl -> if eval b then eval tr else eval fl
```

GADT があると型推論は多くの場合困難になり、しばしば明示的な注釈が必要になります。例えば、``f`` は` `T a -> [a]`` の型も ``T a -> [Int]`` の型も持つことができ、どちらも主要型ではありません。

```haskell
data T :: * -> * where
  T1 :: Int -> T Int
  T2 :: T a

f (T1 n) = [n]
f T2     = []
```

## 種注釈

Haskell の種（即ち型の型）のシステムは、基本の種 ``*``（スター）と関数の種 ``->`` からできています。

```haskell
κ : *
  | κ -> κ
```

```haskell
Int :: *
Maybe :: * -> *
Either :: * -> * -> *
```

このシステムには実はいくつかの拡張があり、それらについては後で触れます。（[種多相](昇格#kind-polymorphism)と[非ボックス型](GHC#unboxed-types)を参照。）しかし、普段のコードで使う種のほとんどは、単純にスターか矢印かで出来ています。

KindSignatures［型注釈］拡張を有効にすると、トップレベルの型シグニチャで明示的に種を指定することが出来、通常の種推論の道筋に脇道を付けることが出来ます。

```haskell
{-# LANGUAGE KindSignatures #-}

id :: forall (a :: *). a -> a
id x = x
```

デフォルトの GADTs宣言に加えて、GADTsの引数を特定の種に制限することもできます。基本的な使い方をしていれば Haskell の型推論器はまあまあ上手く推論できますが、種システムを拡張する、いくつかの型システムの拡張と組み合わせると、種注釈は必須になります。

```haskell
{-# Language GADTs #-}
{-# LANGUAGE KindSignatures #-}

data Term a :: * where
  Lit    :: a -> Term a
  Succ   :: Term Int -> Term Int
  IsZero :: Term Int -> Term Bool
  If     :: Term Bool -> Term a -> Term a -> Term a

data Vec :: * -> * -> * where
  Nil :: Vec n a
  Cons :: a -> Vec n a -> Vec n a

data Fix :: (* -> *) -> * where
  In :: f (Fix f) -> Fix f
```

## Void

Void型は、値を含まない型です。自分とのみ単一化されます。

newtypeラッパを使えば、再帰のせいで値を作ることができない型を作成できます。

```haskell
-- Void :: Void -> Void
newtype Void = Void Void
```

あるいは、``-XEmptyDataDecls``を使えば、同じように値を含まない型を、コンストラクタの無いデータ宣言として作成することもできます。

```haskell
data Void
```

両者とも、含んでいる唯一の項は、``undefined``のような発散する項です。

## 幽霊型

幽霊型とは、型宣言の左辺に現れるけれども、型が含む値による制約を受けていない、型引数のことです。実用上は、型レベルで追加の情報を書き表すための入れ口です。

```haskell
import Data.Void

data Foo tag a = Foo a

combine :: Num a => Foo tag a -> Foo tag a -> Foo tag a
combine (Foo a) (Foo b) = Foo (a+b)

-- 値レベルでは全て同じですが、型レベルでは違います。
a :: Foo () Int
a = Foo 1

b :: Foo t Int
b = Foo 1

c :: Foo Void Int
c = Foo 1

-- () ~ ()
example1 :: Foo () Int
example1 = combine a a

-- t ~ ()
example2 :: Foo () Int
example2 = combine a b

-- t0 ~ t1
example3 :: Foo t Int
example3 = combine b b

-- Couldn't match type `t' with `Void'
example4 :: Foo t Int
example4 = combine b c
```

型変数 ``tag`` は宣言の右辺に現れないということに注目してください。これを使えば、値レベルで明示しなくてよい、型レベルの不変条件を表現できます。型レベルで余分な情報を付加することで、効果的にプログラミングが出来るのです。

参照：

* [Fun with Phantom Types](http://www.researchgate.net/publication/228707929_Fun_with_phantom_types/file/9c960525654760c169.pdf)

## 型等価性

データ型について多機能の言語では、コンストラクタで項の関係を目撃することのできる型を表現できます。例えば、2 つの型の間の命題的相等性を表す項を表現できるようになります。

型``Eql a b``は型``a``と``b``が等しいことの証明であり、単一の``Refl``コンストラクタに対しパターンマッチすることにより、パターンマッチの本体に相等性制約をもたらすことになります。

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ExplicitForAll #-}

-- a ≡ b
data Eql a b where
  Refl :: Eql a a

-- 合同性
-- (f : A → B) {x y} → x ≡ y → f x ≡ f y
cong :: Eql a b -> Eql (f a) (f b)
cong Refl = Refl

-- 対称性
-- {a b : A} → a ≡ b → a ≡ b
sym :: Eql a b -> Eql b a
sym Refl = Refl

-- 推移性
-- {a b c : A} → a ≡ b → b ≡ c → a ≡ c
trans :: Eql a b -> Eql b c -> Eql a c
trans Refl Refl = Refl

-- 与えられた相等性証明により、一方の型を他方の型へと強制変換する。
-- {a b : A} → a ≡ b → a → b
castWith :: Eql a b -> a -> b
castWith Refl = id

-- 自明なもの
a = Refl

b :: forall. Eql () ()
b = Refl
```

GHC 7.8 の時点では、これらのコンストラクタと関数は [Data.Type.Equality](http://hackage.haskell.org/package/base-4.7.0.0/docs/Data-Type-Equality.html) に含まれています。
