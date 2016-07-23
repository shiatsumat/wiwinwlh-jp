# パース処理

## parsec

For parsing in Haskell it is quite common to use a family of libraries known as *Parser Combinators* which let
us write code to generate parsers which themselves looks very similar to the parser grammar itself!

              Combinators
-----------   ------------
``<|>``       The choice operator tries to parse the first argument before proceeding to the second. Can be chained sequentially to a generate a sequence of options.
``many``      Consumes an arbitrary number of patterns matching the given pattern and returns them as a list.
``many1``     Like many but requires at least one match.
``optional``  Optionally parses a given pattern returning it's value as a Maybe.
``try``       Backtracking operator will let us parse ambiguous matching expressions and restart with a different pattern.

There are two styles of writing Parsec, one can choose to write with monads or with applicatives.

```haskell
parseM :: Parser Expr
parseM = do
  a <- identifier
  char '+'
  b <- identifier
  return $ Add a b
```

The same code written with applicatives uses the applicative combinators:

```haskell
-- | Sequential application.
(<*>) :: f (a -> b) -> f a -> f b

-- | Sequence actions, discarding the value of the first argument.
(*>) :: f a -> f b -> f b
(*>) = liftA2 (const id)

-- | Sequence actions, discarding the value of the second argument.
(<*) :: f a -> f b -> f a
(<*) = liftA2 const
```

```haskell
parseA :: Parser Expr
parseA = Add <$> identifier <* char '+' <*> identifier
```

Now for instance if we want to parse simple lambda expressions we can encode the parser logic as compositions
of these combinators which yield the string parser when evaluated under with the ``parse``.

```haskell
import Text.Parsec
import Text.Parsec.String

data Expr
  = Var Char
  | Lam Char Expr
  | App Expr Expr
  deriving Show

lam :: Parser Expr
lam = do
  char '\\'
  n <- letter
  string "->"
  e <- expr
  return $ Lam n e

app :: Parser Expr
app = do
  apps <- many1 term
  return $ foldl1 App apps

var :: Parser Expr
var = do
  n <- letter
  return $ Var n

parens :: Parser Expr -> Parser Expr
parens p = do
  char '('
  e <- p
  char ')'
  return e

term :: Parser Expr
term = var <|> parens expr

expr :: Parser Expr
expr = lam <|> app

decl :: Parser Expr
decl = do
  e <- expr
  eof
  return e

test :: IO ()
test = parseTest decl "\\y->y(\\x->x)y"

main :: IO ()
main = test >>= print
```

## カスタムレクサ

In our previous example lexing pass was not necessary because each lexeme mapped to a sequential collection
of characters in the stream type. If we wanted to extend this parser with a non-trivial set of tokens, then
Parsec provides us with a set of functions for defining lexers and integrating these with the parser
combinators. The simplest example builds on top of the builtin Parsec language definitions which define a set
of most common lexical schemes.

```haskell
haskellDef   :: LanguageDef st
emptyDef     :: LanguageDef st
haskellStyle :: LanguageDef st
javaStyle    :: LanguageDef st
```

For instance we'll build on top of the empty language grammar.

```haskell
import Text.Parsec
import Text.Parsec.Expr
import Text.Parsec.String
import qualified Text.Parsec.Token as Token

lexerStyle :: Token.LanguageDef ()
lexerStyle = Token.LanguageDef
  { Token.commentStart   = "{-"
  , Token.commentEnd     = "-}"
  , Token.commentLine    = "--"
  , Token.nestedComments = True
  , Token.identStart     = letter
  , Token.identLetter    = alphaNum <|> oneOf "_"
  , Token.opStart        = Token.opLetter lexerStyle
  , Token.opLetter       = oneOf "`~!@$%^&*-+=;:<>./?"
  , Token.reservedOpNames= []
  , Token.reservedNames  = ["if", "then", "else", "def"]
  , Token.caseSensitive  = True
  }

lexer :: Token.TokenParser ()
lexer = Token.makeTokenParser lexerStyle

parens :: Parser a -> Parser a
parens = Token.parens lexer

natural :: Parser Integer
natural = Token.natural lexer

identifier :: Parser String
identifier = Token.identifier lexer

reservedOp :: String -> Parser ()
reservedOp = Token.reservedOp lexer

reserved :: String -> Parser ()
reserved = Token.reserved lexer

whiteSpace :: Parser ()
whiteSpace = Token.whiteSpace lexer

comma :: Parser String
comma = Token.comma lexer
```

See: [Text.ParserCombinators.Parsec.Language](http://hackage.haskell.org/package/parsec-3.1.5/docs/Text-ParserCombinators-Parsec-Language.html)

## 単純なパース処理

Putting our lexer and parser together we can write down a more robust parser for our little lambda calculus
syntax.

```haskell
module Parser (parseExpr) where

import Text.Parsec
import Text.Parsec.String (Parser)
import Text.Parsec.Language (haskellStyle)

import qualified Text.Parsec.Expr as Ex
import qualified Text.Parsec.Token as Tok

type Id = String

data Expr
  = Lam Id Expr
  | App Expr Expr
  | Var Id
  | Num Int
  | Op  Binop Expr Expr
  deriving (Show)

data Binop = Add | Sub | Mul deriving Show

lexer :: Tok.TokenParser ()
lexer = Tok.makeTokenParser style
  where ops = ["->","\\","+","*","-","="]
        style = haskellStyle {Tok.reservedOpNames = ops }

reservedOp :: String -> Parser ()
reservedOp = Tok.reservedOp lexer

identifier :: Parser String
identifier = Tok.identifier lexer

parens :: Parser a -> Parser a
parens = Tok.parens lexer

contents :: Parser a -> Parser a
contents p = do
  Tok.whiteSpace lexer
  r <- p
  eof
  return r

natural :: Parser Integer
natural = Tok.natural lexer

variable :: Parser Expr
variable = do
  x <- identifier
  return (Var x)

number :: Parser Expr
number = do
  n <- natural
  return (Num (fromIntegral n))

lambda :: Parser Expr
lambda = do
  reservedOp "\\"
  x <- identifier
  reservedOp "->"
  e <- expr
  return (Lam x e)

aexp :: Parser Expr
aexp =  parens expr
    <|> variable
    <|> number
    <|> lambda

term :: Parser Expr
term = Ex.buildExpressionParser table aexp
  where infixOp x f = Ex.Infix (reservedOp x >> return f)
        table = [[infixOp "*" (Op Mul) Ex.AssocLeft],
                 [infixOp "+" (Op Add) Ex.AssocLeft]]

expr :: Parser Expr
expr = do
  es <- many1 term
  return (foldl1 App es)

parseExpr :: String -> Expr
parseExpr input =
  case parse (contents expr) "<stdin>" input of
    Left err -> error (show err)
    Right ast -> ast

main :: IO ()
main = getLine >>= print . parseExpr >> main
```

Trying it out:

```bash
λ: runhaskell simpleparser.hs
1+2
Op Add (Num 1) (Num 2)

\i -> \x -> x
Lam "i" (Lam "x" (Var "x"))

\s -> \f -> \g -> \x -> f x (g x)
Lam "s" (Lam "f" (Lam "g" (Lam "x" (App (App (Var "f") (Var "x")) (App (Var "g") (Var "x"))))))
```

## 状態遷移のあるパース処理

For a more complex use, consider parser that are internally stateful, for example adding operators that can
defined at parse-time and are dynamically added to the ``expressionParser`` table upon definition.

```haskell
module Main where

import qualified Text.Parsec.Expr as Ex
import qualified Text.Parsec.Token as Tok

import Text.Parsec.Language (haskellStyle)

import Data.List
import Data.Function

import Control.Monad.Identity (Identity)

import Text.Parsec
import qualified Text.Parsec as P

type Name = String

data Expr
  = Var Name
  | Lam Name Expr
  | App Expr Expr
  | Let Name Expr Expr
  | BinOp Name Expr Expr
  | UnOp Name Expr
  deriving (Show)

data Assoc
  = OpLeft
  | OpRight
  | OpNone
  | OpPrefix
  | OpPostfix
  deriving Show

data Decl
  = LetDecl Expr
  | OpDecl OperatorDef
  deriving (Show)

type Op x = Ex.Operator String ParseState Identity x
type Parser a = Parsec String ParseState a
data ParseState = ParseState [OperatorDef] deriving Show

data OperatorDef = OperatorDef {
    oassoc :: Assoc
  , oprec :: Integer
  , otok :: Name
  } deriving Show

lexer :: Tok.GenTokenParser String u Identity
lexer = Tok.makeTokenParser style
  where ops = ["->","\\","+","*","<","=","[","]","_"]
        names = ["let","in","infixl", "infixr", "infix", "postfix", "prefix"]
        style = haskellStyle { Tok.reservedOpNames = ops
                             , Tok.reservedNames = names
                             , Tok.identLetter = alphaNum <|> oneOf "#'_"
                             , Tok.commentLine = "--"
                             }

reserved   = Tok.reserved lexer
reservedOp = Tok.reservedOp lexer
identifier = Tok.identifier lexer
parens     = Tok.parens lexer
brackets   = Tok.brackets lexer
braces     = Tok.braces lexer
commaSep   = Tok.commaSep lexer
semi       = Tok.semi lexer
integer    = Tok.integer lexer
chr        = Tok.charLiteral lexer
str        = Tok.stringLiteral lexer
operator   = Tok.operator lexer

contents :: Parser a -> Parser a
contents p = do
  Tok.whiteSpace lexer
  r <- p
  eof
  return r

expr :: Parser Expr
expr = do
  es <- many1 term
  return (foldl1 App es)

lambda :: Parser Expr
lambda = do
  reservedOp "\\"
  args <- identifier
  reservedOp "->"
  body <- expr
  return $ Lam args body

letin :: Parser Expr
letin = do
  reserved "let"
  x <- identifier
  reservedOp "="
  e1 <- expr
  reserved "in"
  e2 <- expr
  return (Let x e1 e2)

variable :: Parser Expr
variable = do
  x <- identifier
  return (Var x)


addOperator :: OperatorDef -> Parser ()
addOperator a = P.modifyState $ \(ParseState ops) -> ParseState (a : ops)

mkTable :: ParseState -> [[Op Expr]]
mkTable (ParseState ops) =
  map (map toParser) $
    groupBy ((==) `on` oprec) $
      reverse $ sortBy (compare `on` oprec) $ ops

toParser :: OperatorDef -> Op Expr
toParser (OperatorDef ass _ tok) = case ass of
    OpLeft    -> infixOp tok (BinOp tok) (toAssoc ass)
    OpRight   -> infixOp tok (BinOp tok) (toAssoc ass)
    OpNone    -> infixOp tok (BinOp tok) (toAssoc ass)
    OpPrefix  -> prefixOp tok (UnOp tok)
    OpPostfix -> postfixOp tok (UnOp tok)
  where
    toAssoc OpLeft = Ex.AssocLeft
    toAssoc OpRight = Ex.AssocRight
    toAssoc OpNone = Ex.AssocNone
    toAssoc _ = error "no associativity"

infixOp :: String -> (a -> a -> a) -> Ex.Assoc -> Op a
infixOp x f = Ex.Infix (reservedOp x >> return f)

prefixOp :: String -> (a -> a) -> Ex.Operator String u Identity a
prefixOp name f = Ex.Prefix (reservedOp name >> return f)

postfixOp :: String -> (a -> a) -> Ex.Operator String u Identity a
postfixOp name f = Ex.Postfix (reservedOp name >> return f)

term :: Parser Expr
term = do
  tbl <- getState
  let table = mkTable tbl
  Ex.buildExpressionParser table aexp

aexp :: Parser Expr
aexp =  letin
    <|> lambda
    <|> variable
    <|> parens expr

letdecl :: Parser Decl
letdecl = do
  e <- expr
  return $ LetDecl e


opleft :: Parser Decl
opleft = do
  reserved "infixl"
  prec <- integer
  sym <- parens operator
  let op = (OperatorDef OpLeft prec sym)
  addOperator op
  return $ OpDecl op

opright :: Parser Decl
opright = do
  reserved "infixr"
  prec <- integer
  sym <- parens operator
  let op = (OperatorDef OpRight prec sym)
  addOperator op
  return $ OpDecl op

opnone :: Parser Decl
opnone = do
  reserved "infix"
  prec <- integer
  sym <- parens operator
  let op = (OperatorDef OpNone prec sym)
  addOperator op
  return $ OpDecl op

opprefix :: Parser Decl
opprefix = do
  reserved "prefix"
  prec <- integer
  sym <- parens operator
  let op = OperatorDef OpPrefix prec sym
  addOperator op
  return $ OpDecl op

oppostfix :: Parser Decl
oppostfix = do
  reserved "postfix"
  prec <- integer
  sym <- parens operator
  let op = OperatorDef OpPostfix prec sym
  addOperator op
  return $ OpDecl op

decl :: Parser Decl
decl =
    try letdecl
    <|> opleft
    <|> opright
    <|> opnone
    <|> opprefix
    <|> oppostfix

top :: Parser Decl
top = do
  x <- decl
  P.optional semi
  return x


modl :: Parser [Decl]
modl = many top

parseModule :: SourceName -> String -> Either ParseError [Decl]
parseModule filePath = P.runParser (contents modl) (ParseState []) filePath

main :: IO ()
main = do
  input <- readFile "test.in"
  let res = parseModule "<stdin>" input
  case res of
    Left err -> print err
    Right ast -> mapM_ print ast
```

For example input try:

```haskell
infixl 3 ($);
infixr 4 (#);

infix 4 (.);

prefix 10 (-);
postfix 10 (!);

let z = y in a $ a $ (-a)!;
let z = y in a # a # a $ b; let z = y in a # a # a # b;
```

## ジェネリックなパース処理

Previously we defined generic operations for pretty printing and this begs the
question of whether we can write a parser on top of Generics. The answer is
generally yes, so long as there is a direct mapping between the specific lexemes
and sum and products types. Consider the simplest case where we just read off
the names of the constructors using the regular Generics machinery and then
build a Parsec parser terms of them.

```haskell
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Text.Parsec
import Text.Parsec.Text.Lazy
import Control.Applicative ((<*), (<*>), (<$>))
import GHC.Generics

class GParse f where
  gParse :: Parser (f a)

-- Type synonym metadata for constructors
instance (GParse f, Constructor c) => GParse (C1 c f) where
  gParse =
    let con = conName (undefined :: t c f a) in
    (fmap M1 gParse) <* string con

-- Constructor names
instance GParse f => GParse (D1 c f) where
  gParse = fmap M1 gParse

-- XXX
{-instance GParse f => GParse (S1 c f) where-}
  {-gParse = fmap M1 gParse-}

-- Sum types
instance (GParse a, GParse b) => GParse (a :+: b) where
  gParse = try (fmap L1 gParse <|> fmap R1 gParse)

-- Product types
instance (GParse f, GParse g) => GParse (f :*: g) where
  gParse = (:*:) <$> gParse <*> gParse

-- Nullary constructors
instance GParse U1 where
  gParse = return U1

data Scientist
  = Newton
  | Einstein
  | Schrodinger
  | Feynman
  deriving (Show, Generic)

data Musician
  = Vivaldi
  | Bach
  | Mozart
  | Beethoven
  deriving (Show, Generic)

gparse :: (Generic g, GParse (Rep g)) => Parser g
gparse = fmap to gParse

scientist :: Parser Scientist
scientist = gparse

musician :: Parser Musician
musician = gparse
```

```haskell
λ: parseTest parseMusician "Bach"
Bach

λ: parseTest parseScientist "Feynman"
Feynman
```

## attoparsec

Attoparsec is a parser combinator like Parsec but more suited for bulk parsing of large text and binary files
instead of parsing language syntax to ASTs. When written properly Attoparsec parsers can be [extremely
efficient](http://www.serpentine.com/blog/2014/05/31/attoparsec/).

One notable distinction between Parsec and Attoparsec is that backtracking
operator (``try``) is not present and reflects on attoparsec's different
underlying parser model.

For a simple little lambda calculus language we can use attoparsec much in the
same we used parsec:

```haskell
{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -fno-warn-unused-do-bind #-}

import Control.Applicative
import Data.Attoparsec.Text
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.List (foldl1')

data Name
  = Gen Int
  | Name T.Text
  deriving (Eq, Show, Ord)

data Expr
  = Var Name
  | App Expr Expr
  | Lam [Name] Expr
  | Lit Int
  | Prim PrimOp
  deriving (Eq, Show)

data PrimOp
  = Add
  | Sub
  | Mul
  | Div
  deriving (Eq, Show)

data Defn = Defn Name Expr
  deriving (Eq, Show)

name :: Parser Name
name = Name . T.pack <$> many1 letter

num :: Parser Expr
num = Lit <$> signed decimal

var :: Parser Expr
var = Var <$> name

lam :: Parser Expr
lam = do
  string "\\"
  vars <- many1 (skipSpace *> name)
  skipSpace *> string "->"
  body <- expr
  return (Lam vars body)

eparen :: Parser Expr
eparen = char '(' *> expr <* skipSpace <* char ')'

prim :: Parser Expr
prim = Prim <$> (
      char '+' *> return Add
  <|> char '-' *> return Sub
  <|> char '*' *> return Mul
  <|> char '/' *> return Div)

expr :: Parser Expr
expr = foldl1' App <$> many1 (skipSpace *> atom)

atom :: Parser Expr
atom = try lam
    <|> eparen
    <|> prim
    <|> var
    <|> num

def :: Parser Defn
def = do
  skipSpace
  nm <- name
  skipSpace *> char '=' *> skipSpace
  ex <- expr
  skipSpace <* char ';'
  return $ Defn nm ex

file :: T.Text -> Either String [Defn]
file = parseOnly (many def <* skipSpace)

parseFile :: FilePath -> IO (Either T.Text [Defn])
parseFile path = do
  contents <- T.readFile path
  case file contents of
    Left a -> return $ Left (T.pack a)
    Right b -> return $ Right b

main :: IO (Either T.Text [Defn])
main = parseFile "simple.ml"
```

For an example try the above parser with the following simple lambda expression.

```haskell
f = g (x - 1);
g = f (x + 1);
h = \x y -> (f x) + (g y);
```

Attoparsec adapts very well to binary and network protocol style parsing as
well, this is extracted from a small implementation of a distributed consensus
network protocol:

```haskell
{-# LANGUAGE OverloadedStrings #-}

import Control.Monad

import Data.Attoparsec
import Data.Attoparsec.Char8 as A
import Data.ByteString.Char8

data Action
  = Success
  | KeepAlive
  | NoResource
  | Hangup
  | NewLeader
  | Election
  deriving Show

type Sender = ByteString
type Payload = ByteString

data Message = Message
  { action :: Action
  , sender :: Sender
  , payload :: Payload
  } deriving Show

proto :: Parser Message
proto = do
  act  <- paction
  send <- A.takeTill (== '.')
  body <- A.takeTill (A.isSpace)
  endOfLine
  return $ Message act send body

paction :: Parser Action
paction = do
  c <- anyWord8
  case c of
    1  -> return Success
    2  -> return KeepAlive
    3  -> return NoResource
    4  -> return Hangup
    5  -> return NewLeader
    6  -> return Election
    _  -> mzero

main :: IO ()
main = do
  let msgtext = "\x01\x6c\x61\x70\x74\x6f\x70\x2e\x33\x2e\x31\x34\x31\x35\x39\x32\x36\x35\x33\x35\x0A"
  let msg = parseOnly proto msgtext
  print msg
```

See: [Text Parsing Tutorial](https://www.fpcomplete.com/school/starting-with-haskell/libraries-and-frameworks/text-manipulation/attoparsec)
