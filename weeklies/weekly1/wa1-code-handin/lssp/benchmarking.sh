mkdir -p ./test_data/
echo "GENERATING: data for lssp-zeros.fut"
[[ ! -f test_data/lssp-zeros-data-small-interval-small.in ]] && futhark dataset --i32-bounds=-1:1 -b -g '[64000]i32' > test_data/lssp-zeros-data-small-interval-small.in
[[ ! -f test_data/lssp-zeros-data-large-interval-small.in ]] && futhark dataset --i32-bounds=-2040:2040 -b -g '[64000]i32' > test_data/lssp-zeros-data-large-interval-small.in
[[ ! -f test_data/lssp-zeros-data-small-interval-large.in ]] && futhark dataset --i32-bounds=-1:1 -b -g '[64000000]i32' > test_data/lssp-zeros-data-small-interval-large.in
[[ ! -f test_data/lssp-zeros-data-large-interval-large.in ]] && futhark dataset --i32-bounds=-2040:2040 -b -g '[64000000]i32' > test_data/lssp-zeros-data-large-interval-large.in

echo "GENERATING: data for lssp-sorted.fut and lssp-same.fut"
[[ ! -f test_data/lssp-sorted-data-small-interval-small.in ]] && futhark dataset --i32-bounds=1:10 -b -g '[64000]i32' > test_data/lssp-sorted-data-small-interval-small.in
[[ ! -f test_data/lssp-sorted-data-large-interval-small.in ]] && futhark dataset --i32-bounds=-2040:2040 -b -g '[64000]i32' > test_data/lssp-sorted-data-large-interval-small.in
[[ ! -f test_data/lssp-sorted-data-small-interval-large.in ]] && futhark dataset --i32-bounds=1:10 -b -g '[64000000]i32' > test_data/lssp-sorted-data-small-interval-large.in
[[ ! -f test_data/lssp-sorted-data-large-interval-large.in ]] && futhark dataset --i32-bounds=-2040:2040 -b -g '[64000000]i32' > test_data/lssp-sorted-data-large-interval-large.in


echo "BENCHMARKING: lssp-zero"
# for backend in "c"; do # "cuda"; do
for backend in "c" "cuda"; do
    for entry in "main" "mainSeq"; do
        echo "RUNNING: futhark bench --backend=$backend lssp-zeros.fut -e $entry"
        futhark bench --backend=$backend lssp-zeros.fut -e $entry
    done
done

echo "BENCHMARKING: lssp-sorted"
# for backend in "c"; do # "cuda"; do
for backend in "c" "cuda"; do
    for entry in "main" "mainSeq"; do
        echo -e "\nRUNNING: futhark bench --backend=$backend lssp-sorted.fut -e $entry"
        futhark bench --backend=$backend lssp-sorted.fut -e $entry
    done
done

echo "BENCHMARKING: lssp-same"
# for backend in "c"; do # "cuda"; do
for backend in "c" "cuda"; do
    for entry in "main" "mainSeq"; do
        echo -e "\nRUNNING: futhark bench --backend=$backend lssp-same.fut -e $entry"
        futhark bench --backend=$backend lssp-same.fut -e $entry
    done
done
