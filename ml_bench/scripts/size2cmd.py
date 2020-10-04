#!/usr/bin/env python3
import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = "Problem size -> cmd of rocblas-bench"
    parser.add_argument("-m", "--M", type=int, help='M')
    parser.add_argument("-n", "--N", type=int, help='N')
    parser.add_argument("-k", "--K", type=int, help='K')
    parser.add_argument("-bc", "--batch_count", type=int, default=1, help='batch count')
    parser.add_argument("-transA", "--transA", type=str, default='N', help='transpose of A')
    parser.add_argument("-transB", "--transB", type=str, default='N', help='transpose of B')
    parser.add_argument("-lda", "--lda", type=int, help='LDA')
    parser.add_argument("-ldb", "--ldb", type=int, help='LDB')
    parser.add_argument("-ldc", "--ldc", type=int, help='LDC')
    parser.add_argument("-ldd", "--ldd", type=int, help='LDD')
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help='alpha')
    parser.add_argument("-b", "--beta", type=float, default=0.0, help='bata')
    parser.add_argument("-stride_a", "--stride_a", type=int, help='stride A')
    parser.add_argument("-stride_b", "--stride_b", type=int, help='stride B')
    parser.add_argument("-stride_c", "--stride_c", type=int, help='stride C')
    parser.add_argument("-stride_d", "--stride_d", type=int, help='stride D')
    parser.add_argument("-dtype", "--dtype", type=str, default="fp32", help='precision')
    parser.add_argument("-f", "--config", type=str, help='config file path')
    parser.add_argument("-o", "--output", type=str, default='rocblas_bench.csv', help='output')
    args = parser.parse_args()

    if not args.config and (args.M and args.N and args.K):
        import pdb; pdb.set_trace()
        print('No problem sizes specified, please use "--help" for more details.')
        exit(-1)

    if args.config:
        pat = re.compile(r'\[.*\]')
        sizes = re.findall(pat, open(args.config).read())
        sizes = [eval(o) for o in sizes]
        with open(args.output, 'w') as fp:
            for s in sizes:
                args.M, args.N, args.batch_count, args.K = s[0], s[1], s[2], s[3]
                args.ldc = s[4] if len(s) >= 5 else args.M
                args.ldd = s[5] if len(s) >= 6 else args.M
                args.lda = s[6] if len(s) >= 7 else (args.M if args.transA.lower() == 'n' else args.K)
                args.ldb = s[7] if len(s) >= 8 else (args.K if args.transB.lower() == 'n' else args.N)
                if args.batch_count > 1:
                    args.stride_a = args.lda * args.K if args.transA.lower() == 'n' else args.lda * args.M
                    args.stride_b = args.ldb * args.N if args.transA.lower() == 'n' else args.ldb * args.K
                    args.stride_c = args.ldc * args.N
                    args.stride_d = args.ldd * args.N
                    cmd = "./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha 1 --lda {}  --stride_a {} --ldb {} --stride_b {} --beta 0 --ldc {} --stride_c {} --batch_count {}".format(args.transA, args.transB, args.M, args.N, args.K, args.lda, args.stride_a, args.ldb, args.stride_b, args.ldc, args.stride_c, args.batch_count)
                else:
                    cmd = "./rocblas-bench -f gemm -r f32_r --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha 1 --lda {} --ldb {} --beta 0 --ldc {}".format(args.transA, args.transB, args.M, args.N, args.K, args.lda, args.ldb, args.ldc)
                fp.writelines(cmd + '\n')
        print("[Output]: %s" % args.output)
    else:
        args.ldc = args.M
        args.ldd = args.M
        args.lda = args.M if args.transA.lower() == 'n' else args.K
        args.ldb = args.K if args.transB.lower() == 'n' else args.N
        if args.batch_count > 1:
            args.stride_a = args.lda * args.K if args.transA.lower() == 'n' else args.lda * args.M
            args.stride_b = args.ldb * args.N if args.transA.lower() == 'n' else args.ldb * args.K
            args.stride_c = args.ldc * args.N
            args.stride_d = args.ldd * args.N
            cmd = "./rocblas-bench -f gemm_strided_batched -r f32_r --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha 1 --lda {}  --stride_a {} --ldb {} --stride_b {} --beta 0 --ldc {} --stride_c {} --batch_count {}".format(args.transA, args.transB, args.M, args.N, args.K, args.lda, args.stride_a, args.ldb, args.stride_b, args.ldc, args.stride_c, args.batch_count)
        else:
            cmd = "./rocblas-bench -f gemm -r f32_r --transposeA {} --transposeB {} -m {} -n {} -k {} --alpha 1 --lda {} --ldb {} --beta 0 --ldc {}".format(args.transA, args.transB, args.M, args.N, args.K, args.lda, args.ldb, args.ldc)
        fp.writelines(cmd + '\n')
        print(cmd + '\n')

