# <a name="gadts">GADT</a>

## 目次

* [GADT](#gadts2)
* [種注釈](#kind-signatures)
* [Void](#void)
* [幽霊型](#phantom-types)
* [型等価性](#type-equality)

## <a name="gadts2">GADT</a>

*Generalized Algebraic Data types* (GADTs) are an extension to algebraic
datatypes that allow us to qualify the constructors to datatypes with type
equality constraints, allowing a class of types that are not expressible using
vanilla ADTs.

``-XGADTs`` implicitly enables an alternative syntax for datatype declarations (
``-XGADTSyntax`` )  such that the following declarations are equivalent:

```haskell
-- Vanilla
data List a
  = Empty
  | Cons a (List a)

-- GADTSyntax
data List a where
  Empty :: List a
  Cons :: a -> List a -> List a
```

For an example use consider the data type ``Term``, we have a term in which we
``Succ`` which takes a ``Term`` parameterized by ``a`` which span all types.
Problems arise between the clash whether (``a ~ Bool``) or (``a ~ Int``) when
trying to write the evaluator.

```haskell
data Term a
  = Lit a
  | Succ (Term a)
  | IsZero (Term a)

-- can't be well-typed :(
eval (Lit i)      = i
eval (Succ t)     = 1 + eval t
eval (IsZero i)   = eval i == 0
```

And we admit the construction of meaningless terms which forces more error
handling cases.

```haskell
-- This is a valid type.
failure = Succ ( Lit True )
```

Using a GADT we can express the type invariants for our language (i.e. only
type-safe expressions are representable). Pattern matching on this GADTs then
carries type equality constraints without the need for explicit tags.

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

This time around:

```haskell
-- This is rejected at compile-time.
failure = Succ ( Lit True )
```

Explicit equality constraints (``a ~ b``) can be added to a function's context.
For example the following expand out to the same types.


```haskell
f :: a -> a -> (a, a)
f :: (a ~ b) => a -> b -> (a,b)
```

```haskell
(Int ~ Int)  => ...
(a ~ Int)    => ...
(Int ~ a)    => ...
(a ~ b)      => ...
(Int ~ Bool) => ... -- Will not typecheck.
```

This is effectively the implementation detail of what GHC is doing behind the
scenes to implement GADTs ( implicitly passing and threading equality terms
around ). If we wanted we could do the same setup that GHC does just using
equality constraints and existential quantification.

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ExistentialQuantification #-}

-- Using Constraints
data Exp a
  = (a ~ Int) => LitInt a
  | (a ~ Bool) => LitBool a
  | forall b. (b ~ Bool) => If (Exp b) (Exp a) (Exp a)

-- Using GADTs
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

In the presence of GADTs inference becomes intractable in many cases, often
requiring an explicit annotation. For example ``f`` can either have ``T a ->
[a]`` or ``T a -> [Int]`` and neither is principle.

```haskell
data T :: * -> * where
  T1 :: Int -> T Int
  T2 :: T a

f (T1 n) = [n]
f T2     = []
```

## <a name="kind-signatures">種注釈</a>

Haskell's kind system (i.e. the "type of the types") is a system consisting the
single kind ``*`` and an arrow kind ``->``.

```haskell
κ : *
  | κ -> κ
```

```haskell
Int :: *
Maybe :: * -> *
Either :: * -> * -> *
```

There are in fact some extensions to this system that will be covered later ( see:
PolyKinds and Unboxed types in later sections ) but most kinds in everyday code
are simply either stars or arrows.

With the KindSignatures extension enabled we can now annotate top level type
signatures with their explicit kinds, bypassing the normal kind inference
procedures.

```haskell
{-# LANGUAGE KindSignatures #-}

id :: forall (a :: *). a -> a
id x = x
```

On top of default GADT declaration we can also constrain the parameters of the
GADT to specific kinds. For basic usage Haskell's kind inference can deduce this
reasonably well, but combined with some other type system extensions that extend
the kind system this becomes essential.

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

## <a name="void">Void</a>

The Void type is the type with no inhabitants. It unifies only with itself.

Using a newtype wrapper we can create a type where recursion makes it impossible
to construct an inhabitant.

```haskell
-- Void :: Void -> Void
newtype Void = Void Void
```

Or using ``-XEmptyDataDecls`` we can also construct the uninhabited type
equivalently as a data declaration with no constructors.

```haskell
data Void
```

The only inhabitant of both of these types is a diverging term like
(``undefined``).

## <a name="phantom-types">幽霊型</a>

Phantom types are parameters that appear on the left hand side of a type declaration but which are not
constrained by the values of the types inhabitants. They are effectively slots for us to encode additional
information at the type-level.

```haskell
import Data.Void

data Foo tag a = Foo a

combine :: Num a => Foo tag a -> Foo tag a -> Foo tag a
combine (Foo a) (Foo b) = Foo (a+b)

-- All identical at the value level, but differ at the type level.
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

Notice t type variable ``tag`` does not appear in the right hand side of the declaration. Using this allows us
to express invariants at the type-level that need not manifest at the value-level. We're effectively
programming by adding extra information at the type-level.

See: [Fun with Phantom Types](http://www.researchgate.net/publication/228707929_Fun_with_phantom_types/file/9c960525654760c169.pdf)

## <a name="type-equality">型等価性</a>

With a richer language for datatypes we can express terms that witness the
relationship between terms in the constructors, for example we can now express a
term which expresses propositional equality between two types.

The type ``Eql a b`` is a proof that types ``a`` and ``b`` are equal, by pattern
matching on the single ``Refl`` constructor we introduce the equality constraint
into the body of the pattern match.

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ExplicitForAll #-}

-- a ≡ b
data Eql a b where
  Refl :: Eql a a

-- Congruence
-- (f : A → B) {x y} → x ≡ y → f x ≡ f y
cong :: Eql a b -> Eql (f a) (f b)
cong Refl = Refl

-- Symmetry
-- {a b : A} → a ≡ b → a ≡ b
sym :: Eql a b -> Eql b a
sym Refl = Refl

-- Transitivity
-- {a b c : A} → a ≡ b → b ≡ c → a ≡ c
trans :: Eql a b -> Eql b c -> Eql a c
trans Refl Refl = Refl

-- Coerce one type to another given a proof of their equality.
-- {a b : A} → a ≡ b → a → b
castWith :: Eql a b -> a -> b
castWith Refl = id

-- Trivial cases
a :: forall n. Eql n n
a = Refl

b :: forall. Eql () ()
b = Refl
```

As of GHC 7.8 these constructors and functions are included in the Prelude in the
[Data.Type.Equality](http://hackage.haskell.org/package/base-4.7.0.0/docs/Data-Type-Equality.html) module.
