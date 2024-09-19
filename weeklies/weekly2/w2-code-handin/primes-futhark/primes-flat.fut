-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }
-- output @ ref10000000.out

let mkFlagArray 't [m] 
            (aoa_shp: [m]i64) (zero: t)
            (aoa_val: [m]t  ) : []t   =
  let shp_rot = map (\i->if i==0 then 0
                         else aoa_shp[i-1]
                    ) (iota m)
  let shp_scn = scan (+) 0 shp_rot
  let aoa_len = if m == 0 then 0
                else shp_scn[m-1]+aoa_shp[m-1]
  let shp_ind = map2 (\shp ind ->
                       if shp==0 then -1
                       else ind
                     ) aoa_shp shp_scn
  in scatter (replicate aoa_len zero)
             shp_ind aoa_val 

let segmented_scan [n] 't (op: t -> t -> t) (ne: t)
                          (flags: [n]bool) (arr: [n]t) : [n]t =
  let (_, res) = unzip <|
    scan (\(x_flag,x) (y_flag,y) ->
             let fl = x_flag || y_flag
             let vl = if y_flag then y else op x y
             in  (fl, vl)
         ) (false, ne) (zip flags arr)
  in  res


let flatten_map_iota [len] (ns : [len]i64)   =
  let reps = (replicate len true) :> [len]bool
  let flag = mkFlagArray ns false reps
  let vals = map (\f -> if f then 0i64 else 1i64) flag
  in segmented_scan (+) 0 flag vals

let flatten_map_replicate ns vs =
  let (flag_n, flag_v) = zip ns vs |> mkFlagArray ns (0,0i64) |> unzip
  let flag_b = map (\f -> if f == 0 then false else true) flag_n
  in segmented_scan (+) 0 flag_b flag_v

let primesFlat (n : i64) : []i64 =
  let sq_primes   = [2i64, 3i64, 5i64, 7i64]
  let len  = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes
      let flat_size = reduce (+) 0 mult_lens

      --------------------------------------------------------------
      -- The current iteration knows the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the correct `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`.

      -- ORIGINAL:
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite

      -- NORMALIZED (and with mult_len (i.e. mm1's)):
      -- let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes
      -- let flat_size = reduce (+) 0 mult_lens
      -- let composite = map2 (\ p mm1 ->
      --                                 let iot = iota mm1
      --                                 let prim_adjusted = map (+2) iot
      --                                 let multiples = map (\ j -> j * p) prim_adjusted
      --                      ) sq_primes mult_lens

      -- To get the iota out of the map I can use the F(Map(Iota)) flattening transformation.
      -- let flatten_map_iota ns = 
      --   let len = length ns
      --   let flag = mkFlagArray ns 0 (replicate len 1)
      --   let vals = map (\f -> if f != 0 then 0 else 1)
      --   in sgmScan (+) 0 flag vals

      -- now let prim_adjusted can simply become:
      -- let prim_adjusted = map (+2) (flatten_map_iota mult_lens)

      -- now we just need to "extradite" (flatten) the inner map
      -- which again mean replicating each p the number of times mult_lens calls for
      -- in nested form:
      -- map2 (\ p imm -> replicate imm p) sq_primes mult_lens
      -- which we in flattened form can get by.
      -- let flatten_map_replicate ns vs =
      --   let (flag_n, flag_v) = zip ns vs |> mkFlagArray ns (0,0) |> unzip
      --   in sgmScan (0) 0 flag_n flag_v
      -- let sq_primes_replicated = flatten_map_replicate sq_primes mult_lens
      -- where the map then simply becomes
      -- let not_primes = map2 (\ p j -> j * p) prim_adjusted sq_primes_replicated
      
      let sq_primes_replicated = flatten_map_replicate mult_lens sq_primes :> [flat_size]i64
      let prim_adjusted = map (+2) (f-- Primes: Flat-Parallel Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }
-- compiled input { 10000000i64 }
-- output @ ref10000000.out

let mkFlagArray 't [m] 
            (aoa_shp: [m]i64) (zero: t)
            (aoa_val: [m]t  ) : []t   =
  let shp_rot = map (\i->if i==0 then 0
                         else aoa_shp[i-1]
                    ) (iota m)
  let shp_scn = scan (+) 0 shp_rot
  let aoa_len = if m == 0 then 0
                else shp_scn[m-1]+aoa_shp[m-1]
  let shp_ind = map2 (\shp ind ->
                       if shp==0 then -1
                       else ind
                     ) aoa_shp shp_scn
  in scatter (replicate aoa_len zero)
             shp_ind aoa_val 

let segmented_scan [n] 't (op: t -> t -> t) (ne: t)
                          (flags: [n]bool) (arr: [n]t) : [n]t =
  let (_, res) = unzip <|
    scan (\(x_flag,x) (y_flag,y) ->
             let fl = x_flag || y_flag
             let vl = if y_flag then y else op x y
             in  (fl, vl)
         ) (false, ne) (zip flags arr)
  in  res


let flatten_map_iota [len] (ns : [len]i64)   =
  let reps = (replicate len true) :> [len]bool in
  let flag = mkFlagArray ns false reps
  let vals = map (\f -> if f then 0i64 else 1i64) flag
  in segmented_scan (+) 0 flag vals

let flatten_map_replicate ns vs =
  let (flag_n, flag_v) = zip ns vs |> mkFlagArray ns (0,0i64) |> unzip
  let flag_b = map (\f -> if f == 0 then false else true) flag_n
  in segmented_scan (+) 0 flag_b flag_v

let primesFlat (n : i64) : []i64 =
  let sq_primes   = [2i64, 3i64, 5i64, 7i64]
  let len  = 8i64
  let (sq_primes, _) =
    loop (sq_primes, len) while len < n do
      -- this is "len = min n (len*len)" 
      -- but without running out of i64 bounds 
      let len = if n / len < len then n else len*len

      let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes
      let flat_size = reduce (+) 0 mult_lens

      --------------------------------------------------------------
      -- The current iteration knows the primes <= 'len', 
      --  based on which it will compute the primes <= 'len*len'
      -- ToDo: replace the dummy code below with the flat-parallel
      --       code that is equivalent with the nested-parallel one:
      --
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite
      --
      -- Your code should compute the correct `not_primes`.
      -- Please look at the lecture slides L2-Flattening.pdf to find
      --  the normalized nested-parallel version.
      -- Note that the scalar computation `mm1 = (len / p) - 1' has
      --  already been distributed and the result is stored in "mult_lens",
      --  where `p \in sq_primes`.
      -- Also note that `not_primes` has flat length equal to `flat_size`
      --  and the shape of `composite` is `mult_lens`.

      -- ORIGINAL:
      --   let composite = map (\ p -> let mm1 = (len / p) - 1
      --                               in  map (\ j -> j * p ) (map (+2) (iota mm1))
      --                       ) sq_primes
      --   let not_primes = reduce (++) [] composite

      -- NORMALIZED (and with mult_len (i.e. mm1's)):
      -- let mult_lens = map (\ p -> (len / p) - 1 ) sq_primes
      -- let flat_size = reduce (+) 0 mult_lens
      -- let composite = map2 (\ p mm1 ->
      --                                 let iot = iota mm1
      --                                 let prim_adjusted = map (+2) iot
      --                                 let multiples = map (\ j -> j * p) prim_adjusted
      --                      ) sq_primes mult_lens

      -- To get the iota out of the map I can use the F(Map(Iota)) flattening transformation.
      -- let flatten_map_iota ns = 
      --   let len = length ns
      --   let flag = mkFlagArray ns 0 (replicate len 1)
      --   let vals = map (\f -> if f != 0 then 0 else 1)
      --   in sgmScan (+) 0 flag vals

      -- now let prim_adjusted can simply become:
      -- let prim_adjusted = map (+2) (flatten_map_iota mult_lens)

      -- now we just need to "extradite" (flatten) the inner map
      -- which again mean replicating each p the number of times mult_lens calls for
      -- in nested form:
      -- map2 (\ p imm -> replicate imm p) sq_primes mult_lens
      -- which we in flattened form can get by.
      -- let flatten_map_replicate ns vs =
      --   let (flag_n, flag_v) = zip ns vs |> mkFlagArray ns (0,0) |> unzip
      --   in sgmScan (0) 0 flag_n flag_v
      -- let sq_primes_replicated = flatten_map_replicate sq_primes mult_lens
      -- where the map then simply becomes
      -- let not_primes = map2 (\ p j -> j * p) prim_adjusted sq_primes_replicated
      
      let sq_primes_replicated = flatten_map_replicate mult_lens sq_primes :> [flat_size]i64
      let prim_adjusted = map (+2) (flatten_map_iota mult_lens) :> [flat_size]i64
      let not_primes = map2 (\ p j -> j * p) prim_adjusted sq_primes_replicated


      -- let not_primes = replicate flat_size 0

      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

       let zero_array = replicate flat_size 0i8
       let mostly_ones= map (\ x -> if x > 1 then 1i8 else 0i8) (iota (len+1))
       let prime_flags= scatter mostly_ones not_primes zero_array
       let sq_primes = filter (\i-> (i > 1i64) && (i <= n) && (prime_flags[i] > 0i8))
                              (0...len)

       in  (sq_primes, len)

  in sq_primes

-- RUN a big test with:
--   $ futhark cuda primes-flat.fut
--   $ echo "10000000i64" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
-- or simply use futhark bench, i.e.,
--   $ futhark bench --backend=cuda primes-flat.fut
let main (n : i64) : []i64 = primesFlat n
latten_map_iota mult_lens) :> [flat_size]i64
      let not_primes = map2 (\ p j -> j * p) prim_adjusted sq_primes_replicated



      -- let not_primes = replicate flat_size 0

      -- If not_primes is correctly computed, then the remaining
      -- code is correct and will do the job of computing the prime
      -- numbers up to n!
      --------------------------------------------------------------
      --------------------------------------------------------------

       let zero_array = replicate flat_size 0i8
       let mostly_ones= map (\ x -> if x > 1 then 1i8 else 0i8) (iota (len+1))
       let prime_flags= scatter mostly_ones not_primes zero_array
       let sq_primes = filter (\i-> (i > 1i64) && (i <= n) && (prime_flags[i] > 0i8))
                              (0...len)

       in  (sq_primes, len)

  in sq_primes

-- RUN a big test with:
--   $ futhark cuda primes-flat.fut
--   $ echo "10000000i64" | ./primes-flat -t /dev/stderr -r 10 > /dev/null
-- or simply use futhark bench, i.e.,
--   $ futhark bench --backend=cuda primes-flat.fut
let main (n : i64) : []i64 = primesFlat n
