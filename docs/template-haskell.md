# テンプレートハスケル

## 準クオート

Quasiquotation allows us to express "quoted" blocks of syntax that need not necessarily be be the syntax of
the host language, but unlike just writing a giant string it is instead parsed into some AST datatype in the
host language. Notably values from the host languages can be injected into the custom language via
user-definable logic allowing information to flow between the two languages.

In practice quasiquotation can be used to implement custom domain specific languages or integrate with other
general languages entirely via code-generation.

We've already seen how to write a Parsec parser, now let's write a quasiquoter for it.

```haskell
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

module Quasiquote where

import Language.Haskell.TH
import Language.Haskell.TH.Syntax
import Language.Haskell.TH.Quote

import Text.Parsec
import Text.Parsec.String (Parser)
import Text.Parsec.Language (emptyDef)

import qualified Text.Parsec.Expr as Ex
import qualified Text.Parsec.Token as Tok

import Control.Monad.Identity

data Expr
  = Tr
  | Fl
  | Zero
  | Succ Expr
  | Pred Expr
  deriving (Eq, Show)

instance Lift Expr where
  lift Tr         = [| Tr |]
  lift Fl         = [| Tr |]
  lift Zero       = [| Zero |]
  lift (Succ a)   = [| Succ a |]
  lift (Pred a)   = [| Pred a |]

type Op = Ex.Operator String () Identity

lexer :: Tok.TokenParser ()
lexer = Tok.makeTokenParser emptyDef

parens :: Parser a -> Parser a
parens = Tok.parens lexer

reserved :: String -> Parser ()
reserved = Tok.reserved lexer

semiSep :: Parser a -> Parser [a]
semiSep = Tok.semiSep lexer

reservedOp :: String -> Parser ()
reservedOp = Tok.reservedOp lexer

prefixOp :: String -> (a -> a) -> Op a
prefixOp x f = Ex.Prefix (reservedOp x >> return f)

table :: [[Op Expr]]
table = [
    [ prefixOp "succ" Succ
    , prefixOp "pred" Pred
    ]
  ]

expr :: Parser Expr
expr = Ex.buildExpressionParser table factor

true, false, zero :: Parser Expr
true  = reserved "true" >> return Tr
false = reserved "false" >> return Fl
zero  = reservedOp "0" >> return Zero

factor :: Parser Expr
factor =
      true
  <|> false
  <|> zero
  <|> parens expr

contents :: Parser a -> Parser a
contents p = do
  Tok.whiteSpace lexer
  r <- p
  eof
  return r

toplevel :: Parser [Expr]
toplevel = semiSep expr

parseExpr :: String -> Either ParseError Expr
parseExpr s = parse (contents expr) "<stdin>" s

parseToplevel :: String -> Either ParseError [Expr]
parseToplevel s = parse (contents toplevel) "<stdin>" s

calcExpr :: String -> Q Exp
calcExpr str = do
  filename <- loc_filename `fmap` location
  case parse (contents expr) filename str of
    Left err -> error (show err)
    Right tag -> [| tag |]

calc :: QuasiQuoter
calc = QuasiQuoter calcExpr err err err
  where err = error "Only defined for values"
```

Testing it out:

```haskell
{-# LANGUAGE QuasiQuotes #-}

import Quasiquote

a :: Expr
a = [calc|true|]
-- Tr

b :: Expr
b = [calc|succ (succ 0)|]
-- Succ (Succ Zero)

c :: Expr
c = [calc|pred (succ 0)|]
-- Pred (Succ Zero)
```

One extremely important feature is the ability to preserve position information so that errors in the embedded
language can be traced back to the line of the host syntax.

## Language.C.Quote

Of course since we can provide an arbitrary parser for the quoted expression, one might consider embedding the
AST of another language entirely. For example C or CUDA C.

```haskell
hello :: String -> C.Func
hello msg = [cfun|

int main(int argc, const char *argv[])
{
    printf($msg);
    return 0;
}

|]
```

Evaluating this we get back an AST representation of the quoted C program which we can manipulate or print
back out to textual C code using ``ppr`` function.

```haskell
Func
  (DeclSpec [] [] (Tint Nothing))
  (Id "main")
  DeclRoot
  (Params
     [ Param (Just (Id "argc")) (DeclSpec [] [] (Tint Nothing)) DeclRoot
     , Param
         (Just (Id "argv"))
         (DeclSpec [] [ Tconst ] (Tchar Nothing))
         (Array [] NoArraySize (Ptr [] DeclRoot))
     ]
     False)
  [ BlockStm
      (Exp
         (Just
            (FnCall
               (Var (Id "printf"))
               [ Const (StringConst [ "\"Hello Haskell!\"" ] "Hello Haskell!")
               ])))
  , BlockStm (Return (Just (Const (IntConst "0" Signed 0))))
  ]
```

In this example we just spliced in the anti-quoted Haskell string in the printf statement, but we can pass
many other values to and from the quoted expressions including identifiers, numbers, and other quoted
expressions which implement the ``Lift`` type class.

For example now if we wanted programmatically generate the source for a CUDA kernel to run on a GPU we can
switch over the CUDA C dialect to emit the C code.

```haskell
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

import Text.PrettyPrint.Mainland
import qualified Language.C.Syntax as C
import qualified Language.C.Quote.CUDA as Cuda

cuda_fun :: String -> Int -> Float -> C.Func
cuda_fun fn n a = [Cuda.cfun|

__global__ void $id:fn (float *x, float *y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if ( i<$n ) { y[i] = $a*x[i] + y[i]; }
}

|]

cuda_driver :: String -> Int -> C.Func
cuda_driver fn n = [Cuda.cfun|

void driver (float *x, float *y) {
  float *d_x, *d_y;

  cudaMalloc(&d_x, $n*sizeof(float));
  cudaMalloc(&d_y, $n*sizeof(float));

  cudaMemcpy(d_x, x, $n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, $n, cudaMemcpyHostToDevice);

  $id:fn<<<($n+255)/256, 256>>>(d_x, d_y);

  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}

|]

makeKernel :: String -> Float -> Int -> [C.Func]
makeKernel fn a n = [
    cuda_fun fn n a
  , cuda_driver fn n
  ]

main :: IO ()
main = do
  let ker = makeKernel "saxpy" 2 65536
  mapM_ (print . ppr) ker
```

Running this we generate:

```cpp
__global__ void saxpy(float* x, float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 65536) {
        y[i] = 2.0 * x[i] + y[i];
    }
}
int driver(float* x, float* y)
{
    float* d_x, * d_y;

    cudaMalloc(&d_x, 65536 * sizeof(float));
    cudaMalloc(&d_y, 65536 * sizeof(float));
    cudaMemcpy(d_x, x, 65536, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, 65536, cudaMemcpyHostToDevice);
    saxpy<<<(65536 + 255) / 256, 256>>>(d_x, d_y);
    return 0;
}
```

Run the resulting output through ``nvcc -ptx -c`` to get the PTX associated with the outputted code.

## テンプレートハスケルの基本

Of course the most useful case of quasiquotation is the ability to procedurally generate Haskell code itself
from inside of Haskell. The ``template-haskell`` framework provides four entry points for the quotation to
generate various types of Haskell declarations and expressions.

Type         Quasiquoted     Class
------------ --------------  -----------
``Q Exp``    ``[e| ... |]``  expression
``Q Pat``    ``[p| ... |]``  pattern
``Q Type``   ``[t| ... |]``  type
``Q [Dec]``  ``[d| ... |]``  declaration

```haskell
data QuasiQuoter = QuasiQuoter
  { quoteExp  :: String -> Q Exp
  , quotePat  :: String -> Q Pat
  , quoteType :: String -> Q Type
  , quoteDec  :: String -> Q [Dec]
  }
```

The logic evaluating, splicing, and introspecting compile-time values is embedded within the Q monad, which
has a ``runQ`` which can be used to evaluate it's context. These functions of this monad is deeply embedded in
the implementation of GHC.

```haskell
runQ :: Quasi m => Q a -> m a
runIO :: IO a -> Q a
```

Just as before, TemplateHaskell provides the ability to lift Haskell values into the their AST quantities
within the quoted expression using the Lift type class.

```haskell
class Lift t where
  lift :: t -> Q Exp

instance Lift Integer where
  lift x = return (LitE (IntegerL x))

instance Lift Int where
  lift x= return (LitE (IntegerL (fromIntegral x)))

instance Lift Char where
  lift x = return (LitE (CharL x))

instance Lift Bool where
  lift True  = return (ConE trueName)
  lift False = return (ConE falseName)

instance Lift a => Lift (Maybe a) where
  lift Nothing  = return (ConE nothingName)
  lift (Just x) = liftM (ConE justName `AppE`) (lift x)

instance Lift a => Lift [a] where
  lift xs = do { xs' <- mapM lift xs; return (ListE xs') }
```

In many cases Template Haskell can be used interactively to explore the AST form of various Haskell syntax.

```haskell
λ: runQ [e| \x -> x |]
LamE [VarP x_2] (VarE x_2)

λ: runQ [d| data Nat = Z | S Nat |]
[DataD [] Nat_0 [] [NormalC Z_2 [],NormalC S_1 [(NotStrict,ConT Nat_0)]] []]

λ: runQ [p| S (S Z)|]
ConP Singleton.S [ConP Singleton.S [ConP Singleton.Z []]]

λ: runQ [t| Int -> [Int] |]
AppT (AppT ArrowT (ConT GHC.Types.Int)) (AppT ListT (ConT GHC.Types.Int))

λ: let g = $(runQ [| \x -> x |])

λ: g 3
3
```

Using
[Language.Haskell.TH](http://hackage.haskell.org/package/template-haskell-2.4.0.0/docs/Language-Haskell-TH-Syntax.html#t:Dec)
we can piece together Haskell AST element by element but subject to our own custom logic to generate the code.
This can be somewhat painful though as the source-language (called ``HsSyn``) to Haskell is enormous,
consisting of around 100 nodes in it's AST many of which are dependent on the state of language pragmas.

```haskell
-- builds the function (f = \(a,b) -> a)
f :: Q [Dec]
f = do
  let f = mkName "f"
  a <- newName "a"
  b <- newName "b"
  return [ FunD f [ Clause [TupP [VarP a, VarP b]] (NormalB (VarE a)) [] ] ]
```

```haskell
my_id :: a -> a
my_id x = $( [| x |] )

main = print (my_id "Hello Haskell!")
```

As a debugging tool it is useful to be able to dump the reified information out for a given symbol
interactively, to do so there is a simple little hack.

```haskell
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

import Text.Show.Pretty (ppShow)
import Language.Haskell.TH

introspect :: Name -> Q Exp
introspect n = do
  t <- reify n
  runIO $ putStrLn $ ppShow t
  [| return () |]
```

```haskell
λ: $(introspect 'id)
VarI
  GHC.Base.id
  (ForallT
     [ PlainTV a_1627405383 ]
     []
     (AppT (AppT ArrowT (VarT a_1627405383)) (VarT a_1627405383)))
  Nothing
  (Fixity 9 InfixL)


λ: $(introspect ''Maybe)
TyConI
  (DataD
     []
     Data.Maybe.Maybe
     [ PlainTV a_1627399528 ]
     [ NormalC Data.Maybe.Nothing []
     , NormalC Data.Maybe.Just [ ( NotStrict , VarT a_1627399528 ) ]
     ]
     [])
```

```haskell
import Language.Haskell.TH

foo :: Int -> Int
foo x = x + 1

data Bar

fooInfo :: InfoQ
fooInfo = reify 'foo

barInfo :: InfoQ
barInfo = reify ''Bar
```

```haskell
$( [d| data T = T1 | T2 |] )

main = print [T1, T2]
```

Splices are indicated by ``$(f)`` syntax for the expression level and at the toplevel simply by invocation of
the template Haskell function. Running GHC with ``-ddump-splices`` shows our code being spliced in at the
specific location in the AST at compile-time.

```haskell
$(f)

template_haskell_show.hs:1:1: Splicing declarations
    f
  ======>
    template_haskell_show.hs:8:3-10
    f (a_a5bd, b_a5be) = a_a5bd
```

```haskell
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

module Splice where

import Language.Haskell.TH
import Language.Haskell.TH.Syntax

spliceF :: Q [Dec]
spliceF = do
  let f = mkName "f"
  a <- newName "a"
  b <- newName "b"
  return [ FunD f [ Clause [VarP a, VarP b] (NormalB (VarE a)) [] ] ]

spliceG :: Lift a => a -> Q [Dec]
spliceG n = runQ [d| g a = n |]
```

```haskell
{-# LANGUAGE TemplateHaskell #-}

import Splice

spliceF
spliceG "argument"

main = do
  print $ f 1 2
  print $ g ()
```

At the point of the splice all variables and types used must be in scope, so it must appear after their
declarations in the module. As a result we often have to mentally topologically sort our code when using
TemplateHaskell such that declarations are defined in order.

See: [Template Haskell AST](http://hackage.haskell.org/package/template-haskell-2.9.0.0/docs/Language-Haskell-TH.html#t:Exp)

## 反クオート

Extending our quasiquotation from above now that we have TemplateHaskell machinery we can implement the same
class of logic that it uses to pass Haskell values in and pull Haskell values out via pattern matching on
templated expressions.

```haskell
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveDataTypeable #-}

module Antiquote where

import Data.Generics
import Language.Haskell.TH
import Language.Haskell.TH.Quote

import Text.Parsec
import Text.Parsec.String (Parser)
import Text.Parsec.Language (emptyDef)

import qualified Text.Parsec.Expr as Ex
import qualified Text.Parsec.Token as Tok

data Expr
  = Tr
  | Fl
  | Zero
  | Succ Expr
  | Pred Expr
  | Antiquote String
  deriving (Eq, Show, Data, Typeable)

lexer :: Tok.TokenParser ()
lexer = Tok.makeTokenParser emptyDef

parens :: Parser a -> Parser a
parens = Tok.parens lexer

reserved :: String -> Parser ()
reserved = Tok.reserved lexer

identifier :: Parser String
identifier = Tok.identifier lexer

semiSep :: Parser a -> Parser [a]
semiSep = Tok.semiSep lexer

reservedOp :: String -> Parser ()
reservedOp = Tok.reservedOp lexer

oper s f assoc = Ex.Prefix (reservedOp s >> return f)

table = [ oper "succ" Succ Ex.AssocLeft
        , oper "pred" Pred Ex.AssocLeft
        ]

expr :: Parser Expr
expr = Ex.buildExpressionParser [table] factor

true, false, zero :: Parser Expr
true  = reserved "true" >> return Tr
false = reserved "false" >> return Fl
zero  = reservedOp "0" >> return Zero

antiquote :: Parser Expr
antiquote = do
  char '$'
  var <- identifier
  return $ Antiquote var

factor :: Parser Expr
factor = true
      <|> false
      <|> zero
      <|> antiquote
      <|> parens expr

contents :: Parser a -> Parser a
contents p = do
  Tok.whiteSpace lexer
  r <- p
  eof
  return r

parseExpr :: String -> Either ParseError Expr
parseExpr s = parse (contents expr) "<stdin>" s


class Expressible a where
  express :: a -> Expr

instance Expressible Expr where
  express = id

instance Expressible Bool where
  express True = Tr
  express False = Fl

instance Expressible Integer where
  express 0 = Zero
  express n = Succ (express (n - 1))


exprE :: String -> Q Exp
exprE s = do
  filename <- loc_filename `fmap` location
  case parse (contents expr) filename s of
    Left err -> error (show err)
    Right exp -> dataToExpQ (const Nothing `extQ` antiExpr) exp

exprP :: String -> Q Pat
exprP s = do
  filename <- loc_filename `fmap` location
  case parse (contents expr) filename s of
    Left err -> error (show err)
    Right exp -> dataToPatQ (const Nothing `extQ` antiExprPat) exp

-- antiquote RHS
antiExpr :: Expr -> Maybe (Q Exp)
antiExpr (Antiquote v) = Just embed
  where embed = [| express $(varE (mkName v)) |]
antiExpr _ = Nothing

-- antiquote LHS
antiExprPat :: Expr -> Maybe (Q Pat)
antiExprPat (Antiquote v) = Just $ varP (mkName v)
antiExprPat _ = Nothing

mini :: QuasiQuoter
mini = QuasiQuoter exprE exprP undefined undefined
```

```haskell
{-# LANGUAGE QuasiQuotes #-}

import Antiquote

-- extract
a :: Expr -> Expr
a [mini|succ $x|] = x

b :: Expr -> Expr
b [mini|succ $x|] = [mini|pred $x|]

c :: Expressible a => a -> Expr
c x = [mini|succ $x|]

d :: Expr
d = c (8 :: Integer)
-- Succ (Succ (Succ (Succ (Succ (Succ (Succ (Succ Zero)))))))

e :: Expr
e = c True
-- Succ Tr
```

## テンプレート型族

Just like at the value-level we can construct type-level constructions by piecing together their AST.

```haskell
Type          AST
----------    ----------
t1 -> t2      ArrowT `AppT` t2 `AppT` t2
[t]           ListT `AppT` t
(t1,t2)       TupleT 2 `AppT` t1 `AppT` t2
```

For example consider that type-level arithmetic is still somewhat incomplete in GHC 7.6, but there often cases
where the span of typelevel numbers is not full set of integers but is instead some bounded set of numbers. We
can instead define operations with a type-family instead of using an inductive definition ( which often
requires manual proofs ) and simply enumerates the entire domain of arguments to the type-family and maps them
to some result computed at compile-time.

For example the modulus operator would be non-trivial to implement at type-level but instead we can use the
``enumFamily`` function to splice in type-family which simply enumerates all possible pairs of numbers up to a
desired depth.

```haskell
module EnumFamily where
import Language.Haskell.TH

enumFamily :: (Integer -> Integer -> Integer)
           -> Name
           -> Integer
           -> Q [Dec]
enumFamily f bop upper = return decls
  where
    decls = do
      i <- [1..upper]
      j <- [2..upper]
      return $ TySynInstD bop (rhs i j)

    rhs i j = TySynEqn
      [LitT (NumTyLit i), LitT (NumTyLit j)]
      (LitT (NumTyLit (i `f` j)))
```

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TemplateHaskell #-}

import EnumFamily

import Data.Proxy
import GHC.TypeLits

type family Mod (m :: Nat) (n :: Nat) :: Nat
type family Add (m :: Nat) (n :: Nat) :: Nat
type family Pow (m :: Nat) (n :: Nat) :: Nat

enumFamily mod ''Mod 10
enumFamily (+) ''Add 10
enumFamily (^) ''Pow 10

a :: Integer
a = natVal (Proxy :: Proxy (Mod 6 4))
-- 2

b :: Integer
b = natVal (Proxy :: Proxy (Pow 3 (Mod 6 4)))
-- 9

--    enumFamily mod ''Mod 3
--  ======>
--    template_typelevel_splice.hs:7:1-14
--    type instance Mod 2 1 = 0
--    type instance Mod 2 2 = 0
--    type instance Mod 2 3 = 2
--    type instance Mod 3 1 = 0
--    type instance Mod 3 2 = 1
--    type instance Mod 3 3 = 0
--    ...
```

In practice GHC seems fine with enormous type-family declarations although compile-time may
increase a bit as a result.

The singletons library also provides a way to automate this process by letting us write seemingly value-level
declarations inside of a quasiquoter and then promoting the logic to the type-level. For example if we wanted
to write a value-level and type-level map function for our HList this would normally involve quite a bit of
boilerplate, now it can stated very concisely.

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}

import Data.Singletons
import Data.Singletons.TH

$(promote [d|
  map :: (a -> b) -> [a] -> [b]
  map _ [] = []
  map f (x:xs) = f x : map f xs
  |])

infixr 5 :::

data HList (ts :: [ * ]) where
  Nil :: HList '[]
  (:::) :: t -> HList ts -> HList (t ': ts)

-- TypeLevel
-- MapJust :: [*] -> [Maybe *]
type MapJust xs = Map Maybe xs

-- Value Level
-- mapJust :: [a] -> [Maybe a]
mapJust :: HList xs -> HList (MapJust xs)
mapJust Nil = Nil
mapJust (x ::: xs) = (Just x) ::: mapJust xs

type A = [Bool, String , Double , ()]

a :: HList A
a = True ::: "foo" ::: 3.14 ::: () ::: Nil


example1 :: HList (MapJust A)
example1 = mapJust a

-- example1 reduces to example2 when expanded
example2 :: HList ([Maybe Bool, Maybe String , Maybe Double , Maybe ()])
example2 = Just True ::: Just "foo" ::: Just 3.14 ::: Just () ::: Nil
```

## テンプレート型クラス

Probably the most common use of Template Haskell is the automatic generation of type-class instances. Consider
if we wanted to write a simple Pretty printing class for a flat data structure that derived the ppr method in
terms of the names of the constructors in the AST we could write a simple instance.

```haskell
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module Class where

import Language.Haskell.TH

class Pretty a where
  ppr :: a -> String

normalCons :: Con -> Name
normalCons (NormalC n _) = n

getCons :: Info -> [Name]
getCons cons = case cons of
    TyConI (DataD    _ _ _ tcons _) -> map normalCons tcons
    con -> error $ "Can't derive for:" ++ (show con)

pretty :: Name -> Q [Dec]
pretty dt = do
  info <- reify dt
  Just cls <- lookupTypeName "Pretty"
  let datatypeStr = nameBase dt
  let cons = getCons info
  let dtype = mkName (datatypeStr)
  let mkInstance xs =
        InstanceD
        []                              -- Context
        (AppT
          (ConT cls)                    -- Instance
          (ConT dtype))                 -- Head
        [(FunD (mkName "ppr") xs)]      -- Methods
  let methods = map cases cons
  return $ [mkInstance methods]

-- Pattern matches on the ``ppr`` method
cases :: Name -> Clause
cases a = Clause [ConP a []] (NormalB (LitE (StringL (nameBase a)))) []
```

In a separate file invoke the pretty instance at the toplevel, and with ``--ddump-splice`` if we want to view
the spliced class instance.

```haskell
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

import Class

data PlatonicSolid
  = Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

pretty ''PlatonicSolid

main :: IO ()
main = do
  putStrLn (ppr Octahedron)
  putStrLn (ppr Dodecahedron)
```

## テンプレートシングルトン

In the previous discussion about singletons, we introduced quite a bit of boilerplate code to work with the
singletons. This can be partially abated by using Template Haskell to mechanically generate the instances and
classes.

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

module Singleton where

import Text.Read
import Language.Haskell.TH
import Language.Haskell.TH.Quote

data Nat = Z | S Nat

data SNat :: Nat -> * where
  SZero :: SNat Z
  SSucc :: SNat n -> SNat (S n)

-- Quasiquoter for Singletons

sval :: String -> Q Exp
sval str = do
  case readEither str of
    Left err -> fail (show err)
    Right n -> do
      Just suc <- lookupValueName "SSucc"
      Just zer <- lookupValueName "SZero"
      return $ foldr AppE (ConE zer) (replicate n (ConE suc))

stype :: String -> Q Type
stype str = do
  case readEither str of
    Left err -> fail (show err)
    Right n -> do
      Just scon <- lookupTypeName "SNat"
      Just suc <- lookupValueName "S"
      Just zer <- lookupValueName "Z"
      let nat = foldr AppT (PromotedT zer) (replicate n (PromotedT suc))
      return $ AppT (ConT scon) nat

spat :: String -> Q Pat
spat str = do
  case readEither str of
    Left err -> fail (show err)
    Right n -> do
      Just suc <- lookupValueName "SSucc"
      Just zer <- lookupValueName "SZero"
      return $ foldr (\x y -> ConP x [y]) (ConP zer []) (replicate n (suc))

sdecl :: String -> a
sdecl _ = error "Cannot make toplevel declaration for snat."

snat :: QuasiQuoter
snat = QuasiQuoter sval spat stype sdecl
```

Trying it out by splicing code at the expression level, type level and as patterns.

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

import Singleton

zero :: [snat|0|]
zero =  [snat|0|]

one :: [snat|1|]
one =  [snat|1|]

two :: [snat|2|]
two =  [snat|2|]

three :: [snat|3|]
three  = [snat|3|]

test :: SNat a -> Int
test x = case x of
  [snat|0|] -> 0
  [snat|1|] -> 1
  [snat|2|] -> 2
  [snat|3|] -> 3

isZero :: SNat a -> Bool
isZero [snat|0|] = True
isZero _ = False
```

The [singletons](https://hackage.haskell.org/package/singletons) package takes this idea to it's logical
conclusion allow us to toplevel declarations of seemingly regular Haskell syntax with singletons spliced in,
the end result resembles the constructions in a dependently typed language if one squints hard enough.

```haskell
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}

import Data.Singletons
import Data.Singletons.TH

$(singletons [d|
  data Nat = Zero | Succ Nat
    deriving (Eq, Show)

  plus :: Nat -> Nat -> Nat
  plus Zero     n = n
  plus (Succ m) n = Succ (plus m n)

  isEven :: Nat -> Bool
  isEven Zero = True
  isEven (Succ Zero) = False
  isEven (Succ (Succ n)) = isEven n
  |])
```

After template splicing we see that we now that several new constructs in scope:

```haskell
type SNat a = Sing Nat a

type family IsEven a :: Bool
type family Plus a b :: Nat

sIsEven :: Sing Nat t0 -> Sing Bool (IsEven t0)
splus   :: Sing Nat a -> Sing Nat b -> Sing Nat (Plus a b)
```
