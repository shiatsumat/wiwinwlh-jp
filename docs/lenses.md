# Lens

<!-- markdownlint-disable MD033 -->

## はじめに

There are several implementations of note that are mostly compatible but differ
in scope:

* *lens-family-core*
* *fc-labels*
* *data-lens-light*
* *lens*

**<span style="color:red">WARNING</span>: The ``lens`` library is considered by
many Haskellers to be deeply pathological and introduces a needless amount of
complexity. Some care should taken when considering it's use, it is included
here for information only and not as endorsement for it's use. Consider
``lens-family-core`` or ``fclabels`` instead.**

## lensライブラリは使うべき？

No. The ``lens`` library is deeply problematic when considered in the context of
the rest of the Haskell ecosystem and should be avoided. While there are some
good ideas around the general ideas of lenses, the ``lens`` library's
implementation contains an enormous amount of unidiomatic and over-engineered
Haskell code whose marginal utility is grossly outweighed by the sheer weight of
the entire edifice and the mental strain that it forces it on other developers
to deduce the types involved in even the simplest expressions.

lens is effectively a laboratory for a certain set of emerging ideas, it's
idiosyncratic with respect to the rest of the ecosystem.

## ファン・ラーホーフェンのレンズ

At it's core a lens is a form of coupled getter and setter functions as a value under an existential functor.

```haskell
--         +---- a : Type of structure
--         | +-- b : Type of target
--         | |
type Lens' a b = forall f. Functor f => (b -> f b) -> (a -> f a)
```

There are two derivations of van Laarhoven lenses, one that allows polymorphic update and one that is strictly
monomorphic. Let's just consider the monomorphic variant first:

```haskell
type Lens' a b = forall f. Functor f => (b -> f b) -> (a -> f a)

newtype Const x a  = Const { runConst :: x } deriving Functor
newtype Identity a = Identity { runIdentity :: a } deriving Functor

lens :: (a -> b) -> (a -> b -> a) -> Lens' a b
lens getter setter l a = setter a <$> l (getter a)

set :: Lens' a b -> b -> a -> a
set l b = runIdentity . l (const (Identity b))

get :: Lens' a b -> a -> b
get l = runConst . l Const

over :: Lens' a b -> (b -> b) -> a -> a
over l f a = set l (f (get l a)) a
```

```haskell
infixl 1 &
infixr 4 .~
infixr 4 %~
infixr 8 ^.

(&) :: a -> (a -> b) -> b
(&) = flip ($)

(^.) = flip get
(.~) = set
(%~) = over
```

Such that we have:

```haskell
s ^. (lens getter setter)       -- getter s
s  & (lens getter setter) .~ b  -- setter s b
```

**Law 1**

```haskell
get l (set l b a) = b
```

**Law 2**

```haskell
set l (view l a) a = a
```

**Law 3**

```haskell
set l b1 (set l b2 a) = set l b1 a
```

With composition identities:

```haskell
x^.a.b ≡ x^.a^.b
a.b %~ f ≡ a %~ b %~ f

x ^. id ≡ x
id %~ f ≡ f
```

While this may look like a somewhat convoluted way of reinventing record update, consider the types of these
functions align very nicely such Lens themselves compose using the normal ``(.)`` composition, although in the
reverse direction of function composition.

```haskell
f     :: a -> b
g     :: b -> c
g . f :: a -> c

f     :: Lens a b  ~  (b -> f b) -> (a -> f a)
g     :: Lens b c  ~  (c -> f c) -> (b -> f b)
f . g :: Lens a c  ~  (c -> f c) -> (a -> f a)
```

```haskell
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

import Data.Functor

type Lens' a b = forall f. Functor f => (b -> f b) -> (a -> f a)

newtype Const x a  = Const { runConst :: x } deriving Functor
newtype Identity a = Identity { runIdentity :: a } deriving Functor

lens :: (s -> a) -> (s -> a -> s) -> Lens' s a
lens getter setter f a = fmap (setter a) (f (getter a))

set :: Lens' a b -> b -> a -> a
set l b = runIdentity . l (const (Identity b))

view :: Lens' a b -> a -> b
view l = runConst . l Const

over :: Lens' a b -> (b -> b) -> a -> a
over l f a = set l (f (view l a)) a

compose :: Lens' a b -> Lens' b c -> Lens' a c
compose l s = l . s

id' :: Lens' a a
id' = id

infixl 1 &
infixr 4 .~
infixr 4 %~
infixr 8 ^.

(^.) = flip view
(.~) = set
(%~) = over

(&) :: a -> (a -> b) -> b
(&) = flip ($)

(+~), (-~), (*~) :: Num b => Lens' a b -> b -> a -> a
f +~ b = f %~ (+b)
f -~ b = f %~ (subtract b)
f *~ b = f %~ (*b)

-- Usage

data Foo = Foo { _a :: Int } deriving Show
data Bar = Bar { _b :: Foo } deriving Show

a :: Lens' Foo Int
a = lens getter setter
  where
    getter :: Foo -> Int
    getter = _a

    setter :: Foo -> Int -> Foo
    setter = (\f new -> f { _a = new })


b :: Lens' Bar Foo
b = lens getter setter
  where
    getter :: Bar -> Foo
    getter = _b

    setter :: Bar -> Foo -> Bar
    setter = (\f new -> f { _b = new })

foo :: Foo
foo = Foo 3

bar :: Bar
bar = Bar foo

example1 = view a foo
example2 = set a 1 foo
example3 = over a (+1) foo
example4 = view (b `compose` a) bar

example1' = foo  ^. a
example2' = foo  &  a .~ 1
example3' = foo  &  a %~ (+1)
example4' = bar  ^. b . a
```

It turns out that these simple ideas lead to a very rich set of composite combinators that be used to perform
a wide for working with substructure of complex data structures.

Combinator      Description
-------------   -----------------------------
``view``        View a single target or fold the targets of a monoidal quantity.
``set``         Replace target with a value and return updated structure.
``over``        Update targets with a function and return updated structure.
``to``          Construct a retrieval function from an arbitrary Haskell function.
``traverse``    Map each element of a structure to an action and collect results.
``ix``          Target the given index of a generic indexable structure.
``toListOf``    Return a list of the targets.
``firstOf``     Returns ``Just`` the target of a prism or Nothing.

Certain patterns show up so frequently that they warrant their own operators, although they can be expressed
textual terms as well.

| Symbolic | Textual Equivalent | Description |
| :------: | -----------------: | :---------: |
| ``^.``   | ``view``           | Access value of target |
| ``.~``   | ``set``            | Replace target ``x`` |
| ``%~``   | ``over``           | Apply function to target |
| ``+~``   | ``over t (+n)``    | Add to target |
| ``-~``   | ``over t (-n)``    | Subtract to target |
| ``*~``   | ``over t (*n)``    | Multiply to target |
| ``//~``  | ``over t (//n)``   | Divide to target |
| ``^~``   | ``over t (^n)``    | Integral power to target |
| ``^^~``  | ``over t (^^n)``   | Fractional power to target |
| ``||~``  | ``over t (|| p)``  | Logical or to target |
| ``&&~``  | ``over t (&& p)``  | Logical and to target |
| ``<>~``  | ``over t (<> n)``  | Append to a monoidal target |
| ``?~``   | ``set t (Just x)`` | Replace target with ``Just x`` |
| ``^?``   | ``firstOf``        | Return ``Just`` target or ``Nothing`` |
| ``^..``  | ``toListOf``       | View list of targets |

Constructing the lens field types from an arbitrary datatype involves a bit of boilerplate code generation.
But compiles into simple calls which translate the fields of a record into functions involving the ``lens``
function and logic for the getter and the setter.

```haskell
import Control.Lens

data Foo = Foo { _field :: Int }

field :: Lens' Foo Int
field = lens getter setter
  where
    getter :: Foo -> Int
    getter = _field

    setter :: Foo -> Int -> Foo
    setter = (\f new -> f { _field = new })
```

These are pure boilerplate, and Template Haskell can automatically generate these functions using
``makeLenses`` by introspecting the AST at compile-time.

```haskell
{-# LANGUAGE TemplateHaskell #-}

import Control.Lens

data Foo = Foo { _field :: Int } deriving Show
makeLenses ''Foo
```

The simplest usage of lens is simply as a more compositional way of dealing with
record access and updates, shown below in comparison with traditional record
syntax:

```haskell
{-# LANGUAGE TemplateHaskell #-}

import Control.Lens

data Rec = MkRec { _foo :: Int , _bar :: Int } deriving Show
makeLenses ''Rec

x :: Rec
x = MkRec { _foo = 1024, _bar = 1024 }

get1 :: Int
get1 = (_foo x) + (_bar x)

get2 :: Int
get2 = (x ^. foo) + (x ^. bar)

get3 :: Int
get3 = (view foo x) + (view bar x)


set1 :: Rec
set1 = x { _foo = 1, _bar = 2 }

set2 :: Rec
set2 = x & (foo .~ 1) . (bar .~ 2)

set3 :: Rec
set3 = x & (set foo 1) . (set bar 2)
```

This pattern has great utility when it comes when dealing with complex and
deeply nested structures:

```haskell
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE RankNTypes #-}


import Control.Lens
import Control.Lens.TH

data Record1 = Record1
  { _a :: Int
  , _b :: Maybe Record2
  } deriving Show

data Record2 = Record2
  { _c :: String
  , _d :: [Int]
  } deriving Show

makeLenses ''Record1
makeLenses ''Record2

records :: [Record1]
records = [
    Record1 {
      _a = 1,
      _b = Nothing
    },
    Record1 {
      _a = 2,
      _b = Just $ Record2 {
        _c = "Picard",
        _d = [1,2,3]
      }
    },
    Record1 {
      _a = 3,
      _b = Just $ Record2 {
        _c = "Riker",
        _d = [4,5,6]
      }
    },
    Record1 {
      _a = 4,
      _b = Just $ Record2 {
        _c = "Data",
        _d = [7,8,9]
      }
    }
  ]

-- Lens targets
ids     = traverse.a
names   = traverse.b._Just.c
nums    = traverse.b._Just.d
listn n = traverse.b._Just.d.ix n

-- Modify to set all 'id' fields to 0
ex1 :: [Record1]
ex1 = set ids 0 records

-- Return a view of the concatenated 'd' fields for all nested records.
ex2 :: [Int]
ex2 = view nums records
-- [1,2,3,4,5,6,7,8,9]

-- Increment all 'id' fields by 1
ex3 :: [Record1]
ex3 = over ids (+1) records

-- Return a list of all 'c' fields.
ex4 :: [String]
ex4 = toListOf names records
-- ["Picard","Riker","Data"]

-- Return the the second element of all 'd' fields.
ex5 :: [Int]
ex5 = toListOf (listn 2) records
-- [3,6,9]
```

Lens also provides us with an optional dense slurry of operators that expand
into combinations of the core combinators. Many of the operators do have a
[consistent naming
scheme](https://www.fpcomplete.com/school/to-infinity-and-beyond/pick-of-the-week/a-little-lens-starter-tutorial#actually-there-are-a-whole-lot-of-operators-in-lens---over-100).

The sheer number of operators provided by lens is a polarizing for some, but all
of the operators can be written in terms of the textual functions (``set``,
``view``, ``over``, ``at``, ...) and some people prefer to use these instead.

If one buys into lens model, it can serve as a partial foundation to write logic
over a wide variety of data structures and computations and subsume many of the
existing patterns found in the Prelude.

```haskell
{-# LANGUAGE NoMonomorphismRestriction #-}

import Control.Lens
import Numeric.Lens
import Data.Complex.Lens

import Data.Complex
import qualified Data.Map as Map

l :: Num a => a
l = view _1 (100, 200)
-- 100

m :: Num a => (a, a, a)
m = (100,200,200) & _3 %~ (+100)
-- (100,200,300)

n :: Num a => [a]
n = [100,200,300] & traverse +~ 1
-- [101,201,301]

o :: Char
o = "frodo" ^?! ix 3
-- 'd'

p :: Num a => [a]
p = [[1,2,3], [4,5,6]] ^. traverse
-- [1,2,3,4,5,6]

q :: Num a => [a]
q = [1,2,3,4,5] ^. _tail
-- [2,3,4,5]

r :: Num a => [Maybe a]
r = [Just 1, Just 2, Just 3] & traverse._Just +~ 1
-- [Just 2, Just 3, Just 4]

s :: Maybe String
s = Map.fromList [("foo", "bar")] ^. at "foo"
-- Just "bar"

t :: Integral a => Maybe a
t = "1010110" ^? binary
-- Just 86

u :: Complex Float
u = (mkPolar 1 pi/2) & _phase +~ pi
-- 0.5 :+ 8.742278e-8

v :: [Integer]
v = [1..10] ^.. folded.filtered even
-- [2,4,6,8,10]

w :: [Integer]
w = [1, 2, 3, 4] & each . filtered even *~ 10
-- [1, 20, 3, 40]

x :: Num a => Maybe a
x = Left 3 ^? _Left
-- Just 3
```

See:

* [A Little Lens Tutorial](https://www.fpcomplete.com/school/to-infinity-and-beyond/pick-of-the-week/a-little-lens-starter-tutorial)
* [CPS based functional references](http://twanvl.nl/blog/haskell/cps-functional-references)
* [Lens infix operators](https://github.com/quchen/articles/blob/master/lens-infix-operators.md)

## Lens family

The interface for ``lens-family`` is very similar to ``lens`` but with a smaller API and core.

```haskell
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

import Lens.Family
import Lens.Family.TH
import Lens.Family.Stock
import Data.Traversable

data Record1 = Record1
  { _a :: Int
  , _b :: Maybe Record2
  } deriving Show

data Record2 = Record2
  { _c :: String
  , _d :: [Int]
  } deriving Show

mkLenses ''Record1
mkLenses ''Record2

records :: [Record1]
records = [
    Record1 {
      _a = 1,
      _b = Nothing
    },
    Record1 {
      _a = 2,
      _b = Just $ Record2 {
        _c = "Picard",
        _d = [1,2,3]
      }
    },
    Record1 {
      _a = 3,
      _b = Just $ Record2 {
        _c = "Riker",
        _d = [4,5,6]
      }
    },
    Record1 {
      _a = 4,
      _b = Just $ Record2 {
        _c = "Data",
        _d = [7,8,9]
      }
    }
  ]

ids   = traverse.a
names = traverse.b._Just.c
nums  = traverse.b._Just.d

ex1 = set ids 0 records
ex2 = view nums records
ex3 = over ids (+1) records
ex4 = toListOf names records
```

## 多相更新

```haskell
--        +---- a  : Type of input structure
--        | +-- a' : Type of output structure
--        | |
type Lens a a' b b' = forall f. Functor f => (b -> f b') -> (a -> f a')
--             | |
--             | +-- b  : Type of input target
--             +---- b' : Type of output target
```

```haskell
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

import Data.Functor

type Lens a a' b b' = forall f. Functor f => (b -> f b') -> (a -> f a')
type Lens' a b = Lens a a b b

newtype Const x a  = Const { runConst :: x } deriving Functor
newtype Identity a = Identity { runIdentity :: a } deriving Functor

lens :: (a -> b) -> (a -> b' -> a') -> Lens a a' b b'
lens getter setter f a = fmap (setter a) (f (getter a))

set :: Lens a a' b b' -> b' -> a -> a'
set l b = runIdentity . l (const (Identity b))

get :: Lens a a' b b' -> a -> b
get l = runConst . l Const

over :: Lens a a' b b' -> (b -> b') -> a -> a'
over l f a = set l (f (get l a)) a

compose :: Lens a a' b b' -> Lens b b' c c' -> Lens a a' c c'
compose l s = l . s

id' :: Lens a a a a
id' = id

infixl 1 &
infixr 4 .~
infixr 4 %~
infixr 8 ^.

(^.) = flip get
(.~) = set
(%~) = over

(&) :: a -> (a -> b) -> b
(&) = flip ($)

(+~), (-~), (*~) :: Num b => Lens a a b b -> b -> a -> a
f +~ b = f %~ (+b)
f -~ b = f %~ (subtract b)
f *~ b = f %~ (*b)

-- Monomorphic Update
data Foo = Foo { _a :: Int } deriving Show
data Bar = Bar { _b :: Foo } deriving Show

a :: Lens' Foo Int
a = lens getter setter
  where
    getter :: Foo -> Int
    getter = _a

    setter :: Foo -> Int -> Foo
    setter = (\f new -> f { _a = new })

b :: Lens' Bar Foo
b = lens getter setter
  where
    getter :: Bar -> Foo
    getter = _b

    setter :: Bar -> Foo -> Bar
    setter = (\f new -> f { _b = new })

-- Polymorphic Update
data Pair a b = Pair a b deriving Show

pair :: Pair Int Char
pair = Pair 1 'b'

_1 :: Lens (Pair a b) (Pair a' b) a a'
_1 f (Pair a b) = (\x -> Pair x b) <$> f a

_2 :: Lens (Pair a b) (Pair a b') b b'
_2 f (Pair a b) = (\x -> Pair a x) <$> f b

ex1 = pair ^. _1
ex2 = pair ^. _2
ex3 = pair & _1 .~ "a"
ex4 = pair & (_1  %~ (+1))
           . (_2  .~ 1)
```

## プリズム

```haskell
type Prism a a' b b' = forall f. Applicative f => (b -> f b') -> (a -> f a')
```

Just as lenses allow us to manipulate product types, Prisms allow us to manipulate sum types allowing us to
traverse and apply functions over branches of a sum type selectively.

The two libraries ``lens`` and ``lens-family`` disagree on how these structures are defined and which
constraints they carry but both are defined in terms of at least an Applicative instance. A prism instance in
the lens library is constructed via ``prism`` for polymorphic lens ( those which may change a resulting type
parameter) and ``prism'`` for those which are strictly monomorphic. Just as with the Lens instance
``makePrisms`` can be used to abstract away this boilerplate via Template Haskell.

```haskell
import Control.Lens

data Value = I Int
           | D Double
           deriving Show

_I :: Prism' Value Int
_I = prism remit review
  where
    remit :: Int -> Value
    remit a = I a

    review :: Value -> Either Value Int
    review (I a) = Right a
    review a     = Left a

_D :: Prism' Value Double
_D = prism remit review
  where
    remit :: Double -> Value
    remit a = D a

    review :: Value -> Either Value Double
    review (D a) = Right a
    review a     = Left a


test1 :: Maybe Int
test1 = (I 42) ^? _I

test2 :: Value
test2 = 42 ^. re _I

test3 :: Value
test3 = over _I succ (I 2)

test4 :: Value
test4 = over _I succ (D 2.71)
```

```haskell
_just :: Prism (Maybe a) (Maybe b) a b
_just = prism Just $ maybe (Left Nothing) Right

_nothing :: Prism' (Maybe a) ()
_nothing = prism' (const Nothing) $ maybe (Just ()) (const Nothing)

_left :: Prism (Either a c) (Either b c) a b
_left = prism Left $ either Right (Left . Right)

_right :: Prism (Either c a) (Either c b) a b
_right = prism Right $ either (Left . Left) Right
```

In keeping with the past examples, I'll try to derive Prisms from first principles although this is no easy
task as they typically are built on top of machinery in other libraries. This a (very) rough approximation of
how one might do it using ``lens-family-core`` types.

```haskell
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

import Data.Functor
import Data.Monoid

import Control.Applicative
import Data.Traversable

newtype Getting c a = Getting { unGetting :: c }
newtype Setting a = Setting { unSetting :: a }

type LensLike f s t a b = (a -> f b) -> s -> f t

type Lens a a' b b' = forall f. Functor f => LensLike f a a' b b'
type Lens' a b = Lens a a b b

type Prism a a' b b' = forall f. Applicative f => (b -> f b') -> (a -> f a')
type Prism' a b = Prism a a b b

type Setter a a' b b' = LensLike Setting a a' b b'
type Setter' a b = Setter a a b b

type Getter a c = forall r d b. (c -> Getting r d) -> a -> Getting r b

type FoldLike r a a' b b' = LensLike (Getting r) a a' b b'

instance Functor (Getting c) where
  fmap _ (Getting c) = Getting c

instance Monoid c => Applicative (Getting c) where
  pure _ = Getting mempty
  Getting a <*> Getting b = Getting (a `mappend` b)

class Functor f => Phantom f where
  coerce :: f a -> f b

instance Phantom (Getting c) where
  coerce (Getting c) = Getting c

instance Functor Setting where
  fmap f (Setting a) = Setting (f a)

instance Applicative Setting where
  pure = Setting
  Setting f <*> Setting a = Setting (f a)


lens :: (a -> b) -> (a -> b' -> a') -> Lens a a' b b'
lens getter setter f a = fmap (setter a) (f (getter a))

(.~) :: Setter a a' b b' -> b' -> a -> a'
l .~ b = l %~ const b

view :: FoldLike b a a' b b' -> a -> b
view l = unGetting . l Getting

over :: Setter a a' b b' -> (b -> b') -> a -> a'
over l = (l %~)

set :: Setter a a' b b' -> b' -> a -> a'
set = (.~)

(%~) :: Setter a a' b b' -> (b -> b') -> a -> a'
l %~ f = unSetting . l (Setting . f)

compose :: Lens a a' b b' -> Lens b b' c c' -> Lens a a' c c'
compose l s = l . s

id' :: Lens' a a
id' = id

infixl 1 &
infixr 4 .~
infixr 4 %~
infixr 8 ^.

(^.) :: a -> FoldLike b a a' b b' -> b
(^.) = flip view

(&) :: a -> (a -> b) -> b
(&) = flip ($)

(+~), (-~), (*~) :: Num b => Setter' a b -> b -> a -> a
f +~ b = f %~ (+b)
f -~ b = f %~ (subtract b)
f *~ b = f %~ (*b)


infixr 8 ^?
infixr 8 ^..

views :: FoldLike r a a' b b' -> (b -> r) -> a -> r
views l f = unGetting . l (Getting . f)

(^?) :: a -> FoldLike (First b) a a' b b' -> Maybe b
x ^? l = firstOf l x

(^..) :: a -> FoldLike [b] a a' b b' -> [b]
x ^.. l = toListOf l x

toListOf :: FoldLike [b] a a' b b' -> a -> [b]
toListOf l = views l (:[])

firstOf :: FoldLike (First b) a a' b b' -> a -> Maybe b
firstOf l = getFirst . views l (First . Just)

prism :: (b -> t) -> (s -> Either t a) -> Prism s t a b
prism rm rv f a =
  case rv a of
    Right x -> fmap rm (f x)
    Left x  -> pure x

prism' :: (b -> s) -> (s -> Maybe a) -> Prism s s a b
prism' rm rv f a =
  case rv a of
    Just x  -> fmap rm (f x)
    Nothing -> pure a

_just :: Prism (Maybe a) (Maybe b) a b
_just = prism Just $ maybe (Left Nothing) Right

_nothing :: Prism' (Maybe a) ()
_nothing = prism' (const Nothing) $ maybe (Just ()) (const Nothing)

_right :: Prism (Either c a) (Either c b) a b
_right = prism Right $ either (Left . Left) Right

_left :: Prism (Either a c) (Either b c) a b
_left = prism Left $ either Right (Left . Right)

to :: (s -> a) -> Getter s a
to p f = coerce . f . p



pair :: (Int, Char)
pair = (1, 'b')

_1 :: Lens (a, b) (a', b) a a'
_1 f (a, b) = (\x -> (x, b)) <$> f a

_2 :: Lens (a, b) (a, b') b b'
_2 f (a, b) = (\x -> (a, x)) <$> f b

both :: Prism (a, a) (b, b) a b
both f (a, b) = (,) <$> f a <*> f b

ex1 = pair ^. _1
ex2 = pair ^. _2
ex3 = pair & _1 .~ "a"
ex4 = pair & (_1  %~ (+1))
           . (_2  .~ 1)

ex5 = (1, 2) & both .~ 1
ex6 = Just 3 & _just +~ 1
ex7 = (Left 3) ^? _left
ex8 = over traverse (+1) [1..25]

data Value
  = I Int
  | D Double
  deriving Show

_I :: Prism' Value Int
_I = prism remit review
  where
    remit :: Int -> Value
    remit a = I a

    review :: Value -> Either Value Int
    review (I a) = Right a
    review a     = Left a

ex9 :: Maybe Int
ex9 = (I 42) ^? _I

ex10 :: Value
ex10 = over _I succ (I 2)

ex11 :: Value
ex11 = over _I succ (D 2.71)
```

## 状態モナドとzoom

Within the context of the state monad there are a particularly useful set of lens patterns.

* ``use`` - View a target from the state of the State monad.
* ``assign`` - Replace the target within a State monad.
* ``zoom`` - Modify a target of the state with a function and perform it on the global state of the State monad.

So for example if we wanted to write a little physics simulation of the random motion of particles in a box.
We can use the ``zoom`` function to modify the state of our particles in each step of the simulation.

```haskell
{-# LANGUAGE TemplateHaskell #-}

import Control.Lens
import Control.Monad.State
import System.Random

data Vector = Vector
    { _x :: Double
    , _y :: Double
    } deriving (Show)

data Box = Box
    { _particles :: [Particle]
    } deriving (Show)

data Particle = Particle
    { _pos :: Vector
    , _vel :: Vector
    } deriving (Show)

makeLenses ''Box
makeLenses ''Particle
makeLenses ''Vector

step :: StateT Box IO ()
step = zoom (particles.traverse) $ do
    dx <- use (vel.x)
    dy <- use (vel.y)
    pos.x += dx
    pos.y += dy

particle :: IO Particle
particle = do
  vx <- randomIO
  vy <- randomIO
  return $ Particle (Vector 0 0) (Vector vx vy)

simulate :: IO Box
simulate = do
  particles <- replicateM 5 particle
  let simulation = replicateM 5 step
  let box = Box particles
  execStateT simulation box

main :: IO ()
main = simulate >>= print
```

This results in a final state like the following.

```haskell
Box
  { _particles =
      [ Particle
          { _pos =
              Vector { _x = 3.268546939011934 , _y = 4.356638656040016 }
          , _vel =
              Vector { _x = 0.6537093878023869 , _y = 0.8713277312080032 }
          }
      , Particle
          { _pos =
              Vector { _x = 0.5492296641559635 , _y = 0.27244422070641594 }
          , _vel =
              Vector { _x = 0.1098459328311927 , _y = 5.448884414128319e-2 }
          }
      , Particle
          { _pos =
              Vector { _x = 3.961168796078436 , _y = 4.9317543172941765 }
          , _vel =
              Vector { _x = 0.7922337592156872 , _y = 0.9863508634588353 }
          }
      , Particle
          { _pos =
              Vector { _x = 4.821390854065674 , _y = 1.6601909953629823 }
          , _vel =
              Vector { _x = 0.9642781708131349 , _y = 0.33203819907259646 }
          }
      , Particle
          { _pos =
              Vector { _x = 2.6468253761062943 , _y = 2.161403445396069 }
          , _vel =
              Vector { _x = 0.5293650752212589 , _y = 0.4322806890792138 }
          }
      ]
  }
```

## LensとAeson

One of the best showcases for lens is writing transformations over arbitrary JSON structures. For example
consider some sample data related to Kiva loans.

```json
{
   "loans":[
      {
         "id":2930,
         "terms":{
            "local_payments":[
               {
                  "due_date":"2007-02-08T08:00:00Z",
                  "amount":13.75
               },
               {
                  "due_date":"2007-03-08T08:00:00Z",
                  "amount":93.75
               },
               {
                  "due_date":"2007-04-08T07:00:00Z",
                  "amount":43.75
               },
               {
                  "due_date":"2007-05-08T07:00:00Z",
                  "amount":63.75
               },
               {
                  "due_date":"2007-06-08T07:00:00Z",
                  "amount":93.75
               },
               {
                  "due_date":"2007-07-08T05:00:00Z",
                  "amount": null
               },
               {
                  "due_date":"2007-07-08T07:00:00Z",
                  "amount":93.75
               },
               {
                  "due_date":"2007-08-08T07:00:00Z",
                  "amount":93.75
               },
               {
                  "due_date":"2007-09-08T07:00:00Z",
                  "amount":93.75
               }
            ]
          }
      }
   ]
}
```

Then using ``Data.Aeson.Lens`` we can traverse the structure using our lens combinators.

```haskell
{-# LANGUAGE OverloadedStrings #-}

import Control.Lens

import Data.Aeson.Lens
import Data.Aeson (decode, Value)
import Data.ByteString.Lazy as BL

main :: IO ()
main = do
  contents <- BL.readFile "kiva.json"
  let Just json = decode contents :: Maybe Value

  let vals :: [Double]
      vals = json ^.. key "loans"
                    . values
                    . key "terms"
                    . key "local_payments"
                    . values
                    . key "amount"
                    . _Double
  print vals
```

```haskell
[13.75,93.75,43.75,63.75,93.75,93.75,93.75,93.75]
```
