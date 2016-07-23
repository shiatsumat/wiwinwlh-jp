# ストリーミング

## 遅延IO

The problem with using the usual monadic approach to processing data accumulated through IO is that the
Prelude tools require us to manifest large amounts of data in memory all at once before we can even begin
computation.

```haskell
mapM :: Monad m => (a -> m b) -> [a] -> m [b]
sequence :: Monad m => [m a] -> m [a]
```

Reading from the file creates a thunk for the string that forced will then read the file. The problem is then
that this method ties the ordering of IO effects to evaluation order which is difficult to reason about in the
large.

Consider that normally the monad laws ( in the absence of `seq` ) guarantee that these computations should be
identical. But using lazy IO we can construct a degenerate case.

```haskell
import System.IO

main :: IO ()
main = do
  withFile "foo.txt" ReadMode $ \fd -> do
    contents <- hGetContents fd
    print contents
  -- "foo\n"

  contents <- withFile "foo.txt" ReadMode hGetContents
  print contents
  -- ""
```

So what we need is a system to guarantee deterministic resource handling with constant memory usage. To that
end both the Conduits and Pipes libraries solved this problem using different ( though largely equivalent )
approaches.

## pipes

```haskell
await :: Monad m => Pipe a y m a
yield :: Monad m => a -> Pipe x a m ()

(>->) :: Monad m
      => Pipe a b m r
      -> Pipe b c m r
      -> Pipe a c m r

runEffect :: Monad m => Effect m r -> m r
toListM :: Monad m => Producer a m () -> m [a]
```

Pipes is a stream processing library with a strong emphasis on the static semantics of composition. The
simplest usage is to connect "pipe" functions with a ``(>->)`` composition operator, where each component can
``await`` and ``yield`` to push and pull values along the stream.

```haskell
import Pipes
import Pipes.Prelude as P
import Control.Monad
import Control.Monad.Identity

a :: Producer Int Identity ()
a = forM_ [1..10] yield

b :: Pipe Int Int Identity ()
b =  forever $ do
  x <- await
  yield (x*2)
  yield (x*3)
  yield (x*4)

c :: Pipe Int Int Identity ()
c = forever $ do
  x <- await
  if (x `mod` 2) == 0
    then yield x
    else return ()

result :: [Int]
result = P.toList $ a >-> b >-> c
```

For example we could construct a "FizzBuzz" pipe.

```haskell
{-# LANGUAGE MultiWayIf #-}

import Pipes
import qualified Pipes.Prelude as P

count :: Producer Integer IO ()
count = each [1..100]

fizzbuzz :: Pipe Integer String IO ()
fizzbuzz = do
  n <- await
  if | n `mod` 15 == 0 -> yield "FizzBuzz"
     | n `mod` 5  == 0 -> yield "Fizz"
     | n `mod` 3  == 0 -> yield "Buzz"
     | otherwise       -> return ()
  fizzbuzz

main :: IO ()
main = runEffect $ count >-> fizzbuzz >-> P.stdoutLn
```

To continue with the degenerate case we constructed with Lazy IO, consider than we can now compose and sequence
deterministic actions over files without having to worry about effect order.

```haskell
import Pipes
import Pipes.Prelude as P
import System.IO

readF :: FilePath -> Producer String IO ()
readF file = do
    lift $ putStrLn $ "Opened" ++ file
    h <- lift $ openFile file ReadMode
    fromHandle h
    lift $ putStrLn $ "Closed" ++ file
    lift $ hClose h

main :: IO ()
main = runEffect $ readF "foo.txt" >-> P.take 3 >-> stdoutLn
```

This is simple a sampling of the functionality of lens. The documentation for
pipes is extensive and great deal of care has been taken make the library
extremely thorough. ``pipes`` is a shining example of an accessible yet category
theoretic driven design.

See: [Pipes Tutorial](http://hackage.haskell.org/package/pipes-4.1.0/docs/Pipes-Tutorial.html)

## 安全なpipes

```haskell
bracket :: MonadSafe m => Base m a -> (a -> Base m b) -> (a -> m c) -> m c
```

As a motivating example, ZeroMQ is a network messaging library that abstracts over traditional Unix sockets to
a variety of network topologies.  Most notably it isn't designed to guarantee any sort of transactional
guarantees for delivery or recovery in case of errors so it's necessary to design a layer on top of it to
provide the desired behavior at the application layer.

In Haskell we'd like to guarantee that if we're polling on a socket we get messages delivered in a timely
fashion or consider the resource in an error state and recover from it. Using ``pipes-safe`` we can manage the
life cycle of lazy IO resources and can safely handle failures, resource termination and finalization
gracefully. In other languages this kind of logic would be smeared across several places, or put in some
global context and prone to introduce errors and subtle race conditions. Using pipes we instead get a nice
tight abstraction designed exactly to fit this kind of use case.

For instance now we can bracket the ZeroMQ socket creation and finalization within the ``SafeT`` monad
transformer which guarantees that after successful message delivery we execute the pipes function as expected,
or on failure we halt the execution and finalize the socket.

```haskell
import Pipes
import Pipes.Safe
import qualified Pipes.Prelude as P

import System.Timeout (timeout)
import Data.ByteString.Char8
import qualified System.ZMQ as ZMQ

data Opts = Opts
  { _addr    :: String  -- ^ ZMQ socket address
  , _timeout :: Int     -- ^ Time in milliseconds for socket timeout
  }

recvTimeout :: Opts -> ZMQ.Socket a -> Producer ByteString (SafeT IO) ()
recvTimeout opts sock = do
  body <- liftIO $ timeout (_timeout opts) (ZMQ.receive sock [])
  case body of
    Just msg -> do
      liftIO $ ZMQ.send sock msg []
      yield msg
      recvTimeout opts sock
    Nothing  -> liftIO $ print "socket timed out"

collect :: ZMQ.Context
        -> Opts
        -> Producer ByteString (SafeT IO) ()
collect ctx opts = bracket zinit zclose (recvTimeout opts)
  where
    -- Initialize the socket
    zinit = do
      liftIO $ print "waiting for messages"
      sock <- ZMQ.socket ctx ZMQ.Rep
      ZMQ.bind sock (_addr opts)
      return sock

    -- On timeout or completion guarantee the socket get closed.
    zclose sock = do
      liftIO $ print "finalizing"
      ZMQ.close sock

runZmq :: ZMQ.Context -> Opts -> IO ()
runZmq ctx opts = runSafeT $ runEffect $
  collect ctx opts >-> P.take 10 >-> P.print

main :: IO ()
main = do
  ctx <- ZMQ.init 1
  let opts = Opts {_addr = "tcp://127.0.0.1:8000", _timeout = 1000000 }
  runZmq ctx opts
  ZMQ.term ctx
```

## conduit

```haskell
await :: Monad m => ConduitM i o m (Maybe i)
yield :: Monad m => o -> ConduitM i o m ()
($$) :: Monad m => Source m a -> Sink a m b -> m b
(=$) :: Monad m => Conduit a m b -> Sink b m c -> Sink a m c

type Sink i = ConduitM i Void
type Source m o = ConduitM () o m ()
type Conduit i m o = ConduitM i o m ()
```

Conduits are conceptually similar though philosophically different approach to the same problem of constant
space deterministic resource handling for IO resources.

The first initial difference is that await function now returns a ``Maybe`` which allows different handling of
termination. The composition operators are also split into a connecting operator (``$$``) and a fusing
operator (``=$``) for combining Sources and Sink and a Conduit and a Sink respectively.

```haskell
{-# LANGUAGE MultiWayIf #-}

import Data.Conduit
import Control.Monad.Trans
import qualified Data.Conduit.List as CL

source :: Source IO Int
source = CL.sourceList [1..100]

conduit :: Conduit Int IO String
conduit = do
  val <- await
  liftIO $ print val
  case val of
    Nothing -> return ()
    Just n -> do
      if | n `mod` 15 == 0 -> yield "FizzBuzz"
         | n `mod` 5  == 0 -> yield "Fizz"
         | n `mod` 3  == 0 -> yield "Buzz"
         | otherwise       -> return ()
      conduit

sink :: Sink String IO ()
sink = CL.mapM_ putStrLn

main :: IO ()
main = source $$ conduit =$ sink
```

See: [Conduit Overview](https://www.fpcomplete.com/user/snoyberg/library-documentation/conduit-overview)
