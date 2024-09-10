-- Parallel Longest Satisfying Segment
--
-- ==
-- entry: main
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }  
-- output { 
--    9
-- }
-- compiled input {
--    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
-- }  
-- output { 
--    1
-- }
-- compiled input {
--    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
-- }  
-- output { 
--    10
-- }
-- compiled input {
--    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
-- }  
-- output { 
--    12
-- }
-- compiled input @ test_data/lssp-sorted-data-small-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-small-interval-large.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-large.in

-- INPUTS:
-- futhark dataset --i32-bounds=1:10 -b -g '[64000]i32' > test_data/lssp-sorted-data-small-interval-small.in
-- futhark dataset --i32-bounds=-2040:2040 -b -g '[64000]i32' > test_data/lssp-sorted-data-large-interval-small.in
-- futhark dataset --i32-bounds=1:10 -b -g '[64000000]i32' > test_data/lssp-sorted-data-small-interval-large.in
-- futhark dataset --i32-bounds=-2040:2040 -b -g '[64000000]i32' > test_data/lssp-sorted-data-large-interval-large.in


import "lssp"
import "lssp-seq"

type int = i32

entry main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
  -- in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs

-- Sequential Longest Satisfying Segment
--
-- ==
-- entry: mainSeq
-- compiled input @ test_data/lssp-sorted-data-small-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-small-interval-large.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-large.in

entry mainSeq (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
  in  lssp_seq pred1 pred2 xs
  -- in  lssp pred1 pred2 xs

-- TODO:
-- add a couple of larger datasets and automatically benchmark the sequential and parallel version of the code,
-- e.g., by using futhark bench --backend=c …​ and futhark bench --backend=cuda …​, respectively.
-- (Improved sequential runtime --backend=c can be achieved when using the function lssp_seq instead of lssp,
-- but it is not mandatory.) Report the runtimes and the speedup achieved by GPU acceleration.
-- Several ways of integrating datasets directly in the Futhark program
-- are demonstrated in github file HelperCode/Lect-1-LH/mssp.fut