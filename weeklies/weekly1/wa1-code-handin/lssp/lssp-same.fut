-- Parallel Longest Satisfying Segment
--
-- ==
-- entry: main
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }
-- compiled input {
--    [1i32, -2i32, -2i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    2i32
-- }
-- compiled input {
--    [1i32, 2i32, 3i32, 1i32, 2i32, 3i32]
-- }
-- output {
--    1i32
-- }
-- compiled input {
--    [1i32, -1i32, 1i32, 1i32, -1i32, -1i32, 1i32]
-- }
-- output {
--    2i32
-- }
-- compiled input @ test_data/lssp-sorted-data-small-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-small.in
-- compiled input @ test_data/lssp-sorted-data-small-interval-large.in
-- compiled input @ test_data/lssp-sorted-data-large-interval-large.in
import "lssp"
import "lssp-seq"

entry main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
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
entry mainSeq (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
  in  lssp_seq pred1 pred2 xs
  -- in  lssp pred1 pred2 xs

-- TODO:
-- add a couple of larger datasets and automatically benchmark the sequential and parallel version of the code,
-- e.g., by using futhark bench --backend=c …​ and futhark bench --backend=cuda …​, respectively.
-- (Improved sequential runtime --backend=c can be achieved when using the function lssp_seq instead of lssp,
-- but it is not mandatory.) Report the runtimes and the speedup achieved by GPU acceleration.
-- Several ways of integrating datasets directly in the Futhark program
-- are demonstrated in github file HelperCode/Lect-1-LH/mssp.fut