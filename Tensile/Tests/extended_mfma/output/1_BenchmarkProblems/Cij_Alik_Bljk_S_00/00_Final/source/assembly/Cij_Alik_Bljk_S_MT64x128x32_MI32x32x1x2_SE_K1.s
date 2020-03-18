

/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 8, "AMD", "AMDGPU" 
.text
.protected Cij_Alik_Bljk_S_MT64x128x32_MI32x32x1x2_SE_K1
.globl Cij_Alik_Bljk_S_MT64x128x32_MI32x32x1x2_SE_K1
.p2align 8
.type Cij_Alik_Bljk_S_MT64x128x32_MI32x32x1x2_SE_K1,@function
.amdgpu_hsa_kernel Cij_Alik_Bljk_S_MT64x128x32_MI32x32x1x2_SE_K1
Cij_Alik_Bljk_S_MT64x128x32_MI32x32x1x2_SE_K1:
.amd_kernel_code_t
  is_ptr64 = 1
  enable_sgpr_kernarg_segment_ptr = 1
  kernarg_segment_byte_size = 68 // bytes of kern args
  workitem_vgpr_count = 63 // vgprs
  wavefront_sgpr_count = 81 // sgprs
  compute_pgm_rsrc1_vgprs = 15 // floor((63-1)/4)
  compute_pgm_rsrc1_sgprs = 11 // floor((81-1)/8)
  compute_pgm_rsrc2_tidig_comp_cnt = 0 // 1D wg
  compute_pgm_rsrc2_tgid_x_en = 1 // wg.x
  compute_pgm_rsrc2_tgid_y_en = 1 // wg.y
  workgroup_group_segment_byte_size = 24576 // lds bytes
  compute_pgm_rsrc2_user_sgpr = 2 // vcc
  kernarg_segment_alignment = 4
  group_segment_alignment = 4
  private_segment_alignment = 4
.end_amd_kernel_code_t

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 1 x 32 */
/* SubGroup= 64 x 4 */
/* VectorWidth=1 */
/* GlobalLoadVectorWidthA=2, GlobalLoadVectorWidthB=2 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amd_amdgpu_hsa_metadata
Version: [ 1, 0 ]
Kernels:
  - Name: Cij_Alik_Bljk_S_MT64x128x32_MI32x32x1x2_SE_K1
    SymbolName: 'Cij_Alik_Bljk_S_MT64x128x32_MI32x32x1x2_SE_K1@kd'
    Language: OpenCL C
    LanguageVersion: [ 2, 0 ]
    Args:
      - Name:            sizeC
        Size:            8
        Align:           8
        ValueKind:       ByValue
        ValueType:       I64
      - Name:            sizeA
        Size:            8
        Align:           8
        ValueKind:       ByValue
        ValueType:       I64
      - Name:            sizeB
        Size:            8
        Align:           8
        ValueKind:       ByValue
        ValueType:       I64
      - Name:            D
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       F32
        AddrSpaceQual:   Generic
      - Name:            C
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       F32
        AddrSpaceQual:   Generic
      - Name:            A
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       F32
        AddrSpaceQual:   Generic
      - Name:            B
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       F32
        AddrSpaceQual:   Generic
      - Name:            alpha
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       F32
      - Name:            strideD0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            strideC0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            strideA0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            strideA1
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            strideB0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            strideB1
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            SizesFree0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            SizesFree1
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            SizesSum0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            SizesSum1
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            OrigStaggerUIter
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
      - Name:            NumWorkGroups0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            NumWorkGroups1
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            MagicNumberProblemNumGroupTiles0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            GridNumWorkGroups0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            NumFullBlocks
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            WgmRemainder1
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            MagicNumberWgmRemainder1
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            padding
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
    CodeProps:
      KernargSegmentSize: 136
      GroupSegmentFixedSize: 24576
      PrivateSegmentFixedSize: 0
      KernargSegmentAlign:  8
      WavefrontSize:        64
      NumSGPRs:             81
      NumVGPRs:             63
      MaxFlatWorkGroupSize: 256
.end_amd_amdgpu_hsa_metadata

/******************************************/
/* Asm syntax workarounds                 */
/******************************************/
.macro _v_add_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_add_u32 dst:req, src0:req, src1:req, dpp=
   v_add_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_sub_co_u32 dst:req, cc:req, src0:req, src1:req, dpp=
   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_sub_u32 dst:req, src0:req, src1:req, dpp=
   v_sub_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_addc_co_u32 dst:req, ccOut:req, src0:req, ccIn:req, src1:req, dpp=
   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp
.endm

.macro _v_add_lshl_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_add_u32 dst:req, src0:req, src1:req, shiftCnt:req
    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_cmpx_lt_i16 dst, src0, src1=
   v_cmpx_lt_i16 \dst \src0 \src1 
.endm

.macro _v_cmpx_lt_i32 dst, src0, src1=
   v_cmpx_lt_i32 \dst \src0 \src1 
.endm

.macro _v_cmpx_lt_i64 dst, src0, src1=
   v_cmpx_lt_i64 \dst \src0 \src1 
.endm

.macro _v_cmpx_lt_u16 dst, src0, src1=
   v_cmpx_lt_u16 \dst \src0 \src1 
.endm

.macro _v_cmpx_lt_u32 dst, src0, src1=
   v_cmpx_lt_u32 \dst \src0 \src1 
.endm

.macro _v_cmpx_lt_u64 dst, src0, src1=
   v_cmpx_lt_u64 \dst \src0 \src1 
.endm

.macro _v_cmpx_eq_i16 dst, src0, src1=
   v_cmpx_eq_i16 \dst \src0 \src1 
.endm

.macro _v_cmpx_eq_i32 dst, src0, src1=
   v_cmpx_eq_i32 \dst \src0 \src1 
.endm

.macro _v_cmpx_eq_i64 dst, src0, src1=
   v_cmpx_eq_i64 \dst \src0 \src1 
.endm

.macro _v_cmpx_eq_u16 dst, src0, src1=
   v_cmpx_eq_u16 \dst \src0 \src1 
.endm

.macro _v_cmpx_eq_u32 dst, src0, src1=
   v_cmpx_eq_u32 \dst \src0 \src1 
.endm

.macro _v_cmpx_eq_u64 dst, src0, src1=
   v_cmpx_eq_u64 \dst \src0 \src1 
.endm

.macro _v_cmpx_le_i16 dst, src0, src1=
   v_cmpx_le_i16 \dst \src0 \src1 
.endm

.macro _v_cmpx_le_i32 dst, src0, src1=
   v_cmpx_le_i32 \dst \src0 \src1 
.endm

.macro _v_cmpx_le_i64 dst, src0, src1=
   v_cmpx_le_i64 \dst \src0 \src1 
.endm

.macro _v_cmpx_le_u16 dst, src0, src1=
   v_cmpx_le_u16 \dst \src0 \src1 
.endm

.macro _v_cmpx_le_u32 dst, src0, src1=
   v_cmpx_le_u32 \dst \src0 \src1 
.endm

.macro _v_cmpx_le_u64 dst, src0, src1=
   v_cmpx_le_u64 \dst \src0 \src1 
.endm

.macro _v_cmpx_gt_i16 dst, src0, src1=
   v_cmpx_gt_i16 \dst \src0 \src1 
.endm

.macro _v_cmpx_gt_i32 dst, src0, src1=
   v_cmpx_gt_i32 \dst \src0 \src1 
.endm

.macro _v_cmpx_gt_i64 dst, src0, src1=
   v_cmpx_gt_i64 \dst \src0 \src1 
.endm

.macro _v_cmpx_gt_u16 dst, src0, src1=
   v_cmpx_gt_u16 \dst \src0 \src1 
.endm

.macro _v_cmpx_gt_u32 dst, src0, src1=
   v_cmpx_gt_u32 \dst \src0 \src1 
.endm

.macro _v_cmpx_gt_u64 dst, src0, src1=
   v_cmpx_gt_u64 \dst \src0 \src1 
.endm

.macro _v_cmpx_lg_i16 dst, src0, src1=
   v_cmpx_lg_i16 \dst \src0 \src1 
.endm

.macro _v_cmpx_lg_i32 dst, src0, src1=
   v_cmpx_lg_i32 \dst \src0 \src1 
.endm

.macro _v_cmpx_lg_i64 dst, src0, src1=
   v_cmpx_lg_i64 \dst \src0 \src1 
.endm

.macro _v_cmpx_lg_u16 dst, src0, src1=
   v_cmpx_lg_u16 \dst \src0 \src1 
.endm

.macro _v_cmpx_lg_u32 dst, src0, src1=
   v_cmpx_lg_u32 \dst \src0 \src1 
.endm

.macro _v_cmpx_lg_u64 dst, src0, src1=
   v_cmpx_lg_u64 \dst \src0 \src1 
.endm

.macro _v_cmpx_ge_i16 dst, src0, src1=
   v_cmpx_ge_i16 \dst \src0 \src1 
.endm

.macro _v_cmpx_ge_i32 dst, src0, src1=
   v_cmpx_ge_i32 \dst \src0 \src1 
.endm

.macro _v_cmpx_ge_i64 dst, src0, src1=
   v_cmpx_ge_i64 \dst \src0 \src1 
.endm

.macro _v_cmpx_ge_u16 dst, src0, src1=
   v_cmpx_ge_u16 \dst \src0 \src1 
.endm

.macro _v_cmpx_ge_u32 dst, src0, src1=
   v_cmpx_ge_u32 \dst \src0 \src1 
.endm

.macro _v_cmpx_ge_u64 dst, src0, src1=
   v_cmpx_ge_u64 \dst \src0 \src1 
.endm

.macro _v_cmpx_o_i16 dst, src0, src1=
   v_cmpx_o_i16 \dst \src0 \src1 
.endm

.macro _v_cmpx_o_i32 dst, src0, src1=
   v_cmpx_o_i32 \dst \src0 \src1 
.endm

.macro _v_cmpx_o_i64 dst, src0, src1=
   v_cmpx_o_i64 \dst \src0 \src1 
.endm

.macro _v_cmpx_o_u16 dst, src0, src1=
   v_cmpx_o_u16 \dst \src0 \src1 
.endm

.macro _v_cmpx_o_u32 dst, src0, src1=
   v_cmpx_o_u32 \dst \src0 \src1 
.endm

.macro _v_cmpx_o_u64 dst, src0, src1=
   v_cmpx_o_u64 \dst \src0 \src1 
.endm

.macro _v_cmpx_u_i16 dst, src0, src1=
   v_cmpx_u_i16 \dst \src0 \src1 
.endm

.macro _v_cmpx_u_i32 dst, src0, src1=
   v_cmpx_u_i32 \dst \src0 \src1 
.endm

.macro _v_cmpx_u_i64 dst, src0, src1=
   v_cmpx_u_i64 \dst \src0 \src1 
.endm

.macro _v_cmpx_u_u16 dst, src0, src1=
   v_cmpx_u_u16 \dst \src0 \src1 
.endm

.macro _v_cmpx_u_u32 dst, src0, src1=
   v_cmpx_u_u32 \dst \src0 \src1 
.endm

.macro _v_cmpx_u_u64 dst, src0, src1=
   v_cmpx_u_u64 \dst \src0 \src1 
.endm

/******************************************/
/* Magic div and mod functions            */
/******************************************/
.macro V_MAGIC_DIV dstIdx:req, dividend:req, magicNumber:req, magicShift:req, magicA:req
    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicA
    v_add_u32 v[\dstIdx+0], v[\dstIdx+0], v[\dstIdx+1]
    v_lshrrev_b32 v[\dstIdx+0], \magicShift, v[\dstIdx+0]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 32
.set vgprG2LA, 32
.set vgprValuB_X0_I0, 40
.set vgprG2LB, 40
.set vgprLocalWriteAddrA, 56
.set vgprLocalWriteAddrB, 57
.set vgprGlobalReadOffsetA, 58
.set vgprGlobalReadOffsetB, 59
.set vgprLocalReadAddrA, 60
.set vgprLocalReadAddrB, 61
.set vgprSerial, 62
/* Num VGPR=63 */

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprNumWorkGroups0, 4
.set sgprNumWorkGroups1, 5
.set sgprSrdA, 8
.set sgprSrdB, 12
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprTensor2dSizeA, 24
.set sgprTensor2dSizeB, 26
.set sgprSaveExecMask, 28
.set sgprGSUSumIdx, 30
.set sgprAddressD, 32
.set sgprAddressC, 34
.set sgprStridesC, 36
.set sgprAlpha, 37
.set sgprSizesFree, 38
.set sgprSizesSum, 40
.set sgprLoopCounterK, 42
.set sgprLoopCounterL, 43
.set sgprOrigLoopCounter, 44
.set sgprStridesA, 45
.set sgprStridesB, 47
.set sgprAddressA, 49
.set sgprAddressB, 51
.set sgprShadowLimitA, 54
.set sgprShadowLimitB, 56
.set sgprNumFullBlocks, 58
.set sgprWgmRemainder1, 59
.set sgprMagicNumberWgmRemainder1, 60
.set sgprGlobalReadIncsA, 61
.set sgprGlobalReadIncsB, 63
.set sgprScalarGlobalReadOffsetA, 65
.set sgprScalarGlobalReadOffsetB, 68
/* max SGPR=81 */

/* Size Assignments */
.set sgprSizeI, sgprSizesFree+0
.set sgprSizeJ, sgprSizesFree+1
.set sgprSizeK, sgprSizesSum+0
.set sgprSizeL, sgprSizesSum+1

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesC+0
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set DepthU, 32
.set GSU, 3
.set BpeA, 4
.set BpeALog2, 2
.set BpeB, 4
.set BpeBLog2, 2
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 2
.set SrdShiftLeftB, 2
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0x80000000
.set BufferOOB, 0x80000000

/******************************************/
/* Bits 127:96 of SRD.                    */
/* hex: 0x00020000                        */
/* dst_sel_x (3b): 0                      */
/* dst_sel_y (3b): 0                      */
/* dst_sel_z (3b): 0                      */
/* dst_sel_w (3b): 0                      */
/* num_format (3b): 0                     */
/* data_format (4b): 4                    */
/* user_vm_enable (1b): 0                 */
/* user_vm_mode (1b): 0                   */
/* index_stride (2b): 0                   */
/* add_tid_enable (1b): 0                 */
/* _unusedA (3b): 0                       */
/* nv (1b): 0                             */
/* _unusedB (2b): 0                       */
/* type (2b): 0                           */
/******************************************/
.set Srd127_96, 0x00020000

/* Global Offset A */
.macro GLOBAL_OFFSET_A vgprAddr:req vgprOffsetL:req vgprOffset0I:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideA0I], v[\vgprOffset0I] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/* Global Offset B */
.macro GLOBAL_OFFSET_B vgprAddr:req vgprOffsetL:req vgprOffset1J:req vgprTmp:req
v_mul_lo_u32 v[\vgprTmp+0], s[sgprStrideB1J], v[\vgprOffset1J] // mul d1 lower
_v_add_co_u32 v[\vgprAddr+0], vcc, v[\vgprOffsetL], v[\vgprTmp+0] // accumulate K lower
_v_add_u32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]     // add prepad for pointer shift
v_lshlrev_b32 v[\vgprAddr+0], 0x2, v[\vgprAddr+0]  // offset *= bytes/element
.endm

/******************************************/
/* Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor; */
/******************************************/
.macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp
v_cvt_f32_u32 v[\vQuotient], v[\vDivisor]          // 
v_rcp_f32 v[\vQuotient], v[\vQuotient]             // 
v_mul_f32 v[\vQuotient], 0x4f800000, v[\vQuotient] // 
v_cvt_u32_f32 v[\vQuotient], v[\vQuotient]         // 
v_mul_lo_u32 v[\vRemainder], v[\vDivisor], v[\vQuotient] // 
v_mul_hi_u32 v[\vTmp0], v[\vDivisor], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp1], vcc, 0x0, v[\vRemainder]  // 
v_cmp_ne_i32 s[\sTmp:\sTmp+1], 0x0, v[\vTmp0]      // 
v_cndmask_b32 v[\vRemainder], v[\vTmp1], v[\vRemainder], s[\sTmp:\sTmp+1] // 
v_mul_hi_u32 v[\vRemainder], v[\vRemainder], v[\vQuotient] // 
_v_sub_co_u32 v[\vTmp0], vcc, v[\vQuotient], v[\vRemainder] // 
_v_add_co_u32 v[\vQuotient], vcc, v[\vQuotient], v[\vRemainder] // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vTmp0], s[\sTmp:\sTmp+1] // 
v_mul_hi_u32 v[\vQuotient], v[\vQuotient], v[\vDividend] // 
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vTmp0], vcc, v[\vDividend], v[\vRemainder] // 
v_cmp_ge_u32 s[\sTmp:\sTmp+1], v[\vDividend], v[\vRemainder] // 
_v_add_co_u32 v[\vRemainder], vcc, 0x1, v[\vQuotient] // 
_v_add_co_u32 v[\vTmp1], vcc, -1, v[\vQuotient]    // 
v_cmp_le_u32 vcc, v[\vDivisor], v[\vTmp0]          // 
s_and_b64 vcc, s[\sTmp:\sTmp+1], vcc               // 
v_cndmask_b32 v[\vQuotient], v[\vQuotient], v[\vRemainder], vcc // 
v_cndmask_b32 v[\vQuotient], v[\vTmp1], v[\vQuotient], s[\sTmp:\sTmp+1] // 
v_cmp_ne_i32 vcc, 0x0, v[\vDivisor]                // 
v_cndmask_b32 v[\vQuotient], -1, v[\vQuotient], vcc // final result
v_mul_lo_u32 v[\vRemainder], v[\vQuotient], v[\vDivisor] // 
_v_sub_co_u32 v[\vRemainder], vcc, v[\vDividend], v[\vRemainder] // final result
.endm



/******************************************/
/* Allocate Resources                     */
/******************************************/

s_mov_b32 m0, 0x6000                               // LDS clamp at 24576 bytes
v_mov_b32 v[vgprSerial], v0                        // thread serial id

/* Load Kernel Args */
s_load_dword s[sgprTensor2dSizeA+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x8 // 
s_load_dword s[sgprTensor2dSizeA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0xc // 
s_load_dword s[sgprTensor2dSizeB+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x10 // 
s_load_dword s[sgprTensor2dSizeB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x14 // 
s_load_dword s[sgprAddressD], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x18 // 
s_load_dword s[sgprAddressD+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x1c // 
s_load_dword s[sgprAddressC], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x20 // 
s_load_dword s[sgprAddressC+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x24 // 
s_load_dword s[sgprAddressA], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x28 // 
s_load_dword s[sgprAddressA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x2c // 
s_load_dword s[sgprAddressB], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x30 // 
s_load_dword s[sgprAddressB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x34 // 
s_load_dword s[sgprAlpha], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x38 // 
s_load_dword s[sgprStridesC+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x40 // 
s_load_dword s[sgprStridesA+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x44 // 
s_load_dword s[sgprStridesA+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x48 // 
s_load_dword s[sgprStridesB+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x4c // 
s_load_dword s[sgprStridesB+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x50 // 
s_load_dword s[sgprSizesFree+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x54 // 
s_load_dword s[sgprSizesFree+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x58 // 
s_load_dword s[sgprSizesSum+0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x5c // 
s_load_dword s[sgprSizesSum+1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x60 // 
s_load_dword s[sgprNumWorkGroups0], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x68 // 
s_load_dword s[sgprNumWorkGroups1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x6c // 
s_load_dword s[sgprNumFullBlocks], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x78 // 
s_load_dword s[sgprWgmRemainder1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x7c // 
s_load_dword s[sgprMagicNumberWgmRemainder1], s[sgprKernArgAddress:sgprKernArgAddress+1], 0x80 // 
s_waitcnt lgkmcnt(0)                               // wait for 132 bytes of kern args


/******************************************/
/* Local Read Addresses                   */
/******************************************/


/* local read addresses: tile assignments a */

/*lr0I = serial % SG0I*/
v_lshrrev_b32 v0, 6, v[vgprSerial]                 // vectorStaticDiv: v0 = v[vgprSerial] / 64
v_and_b32 v1, 63, v[vgprSerial]                    // vectorStaticDiv: v1 = v[vgprSerial] % 64


/* local read addresses: tile assignments b */

/*lr1J = (serial / SG1J) % SG1J*/
v_lshrrev_b32 v2, 5, v[vgprSerial]                 // vectorStaticDiv: v2 = v[vgprSerial] / 32
v_and_b32 v3, 31, v[vgprSerial]                    // vectorStaticDiv: v3 = v[vgprSerial] % 32


/* local read addresses: final offsets a */

v_lshrrev_b32 v0, 8, v[vgprSerial]                 // vectorStaticDiv: v0 = v[vgprSerial] / 256
v_and_b32 v2, 255, v[vgprSerial]                   // vectorStaticDiv: v2 = v[vgprSerial] % 256
s_mov_b32 s75, 0x40                                // MT0+PAD
v_mul_lo_u32 v0, s75, v0                           // sgid=sgid*(MT0+PAD)
_v_add_lshl_u32 v[vgprLocalReadAddrA], v0, v1, 0x2 // o = (lroA*VW+sgid*MT0)*bpe


/* local read addresses: final offsets b */

v_lshrrev_b32 v0, 6, v[vgprSerial]                 // vectorStaticDiv: v0 = v[vgprSerial] / 64
v_and_b32 v1, 63, v[vgprSerial]                    // vectorStaticDiv: v1 = v[vgprSerial] % 64
s_mov_b32 s75, 0x20                                // MT1+PAD
v_mul_lo_u32 v0, s75, v0                           // sgid=sgid*(MT1+PAD)
_v_add_lshl_u32 v[vgprLocalReadAddrB], v0, v3, 0x2 // o = (lroB*VW+sgid*MT1)*bpe


/* local read addresses: declare addresses a */

/* N/A */


/* local read addresses: declare addresses b */

_v_add_co_u32 v[vgprLocalReadAddrB+0], vcc, 0x2000, v[vgprLocalReadAddrB+0] //  += LdsOffsetB (lower)



/******************************************/
/* Begin setupNewTile                     */
/******************************************/


/* global read addresses: work-group */

/* graWorkGroup mapping */
// GSU-not-WGMapRR :nwg1 = (size1J + MT1J - 1) / MT1J;
s_mov_b32 s78, s[sgprWorkGroup1]                   // copying for divisor
s_mov_b32 s77, 0x0                                 // STATIC_DIV: divisior=3
s_mul_i32 s76, 0xaaaa, s78                         // tmp1 = dividend * magic hi
s_lshl_b64 s[76:77], s[76:77], 0x10                // left shift 16 bits
s_mul_i32 s[sgprWorkGroup1], s78, 0xaaab           // tmp0 = dividend * magic lo
s_add_u32 s76, s[sgprWorkGroup1], s76              // add lo
s_addc_u32 s77, s77, 0x0                           // add hi
s_lshr_b64 s[76:77], s[76:77], 0x21                // tmp1 = (dividend * magic) << shift
s_mov_b32 s[sgprWorkGroup1], s76                   // quotient
s_mul_i32 s76, s[sgprWorkGroup1], 0x3              // quotient*divisor
s_sub_u32 s[sgprGSUSumIdx], s78, s76               // rReg = dividend - quotient*divisor
s_mov_b32 s79, 0x10000001L                         // magic number for WGM==8
s_mul_hi_u32 s77, s[sgprWorkGroup1], s79           // s_magic mul
s_mul_i32 s76, s[sgprWorkGroup1], s79              // s_magic mul
s_lshr_b64 s[76:77], s[76:77], 31                  // sMagicDiv
s_mul_i32 s77, s76, 8                              // quotient * non-magic divisor
s_sub_u32 s77, s[sgprWorkGroup1], s77              // WorkGroup1=remainder
s_mul_i32 s77, s77, s[sgprNumWorkGroups0]          // (wg1 % WGM)*nwg0
s_add_u32 s77, s77, s[sgprWorkGroup0]              // wgSerial = wg0 + (wg1 % WGM)*nwg0
s_cmp_ge_u32 s76, s[sgprNumFullBlocks]             // blockId >= numFullBlocks ?
s_cmov_b32 s79, s[sgprMagicNumberWgmRemainder1]    // 
s_cselect_b32 s78, s[sgprWgmRemainder1], 8         // 
s_mul_hi_u32 s3, s77, s79                          // s_magic mul
s_mul_i32 s2, s77, s79                             // s_magic mul
s_lshr_b64 s[2:3], s[2:3], 31                      // sMagicDiv
s_mul_i32 s[sgprWorkGroup1], s[sgprWorkGroup0], s78 // quotient * non-magic divisor
s_sub_u32 s[sgprWorkGroup1], s77, s[sgprWorkGroup1] // WorkGroup1=remainder
s_mul_i32 s76, s76, 8                              // blockId * WGM
s_add_u32 s[sgprWorkGroup1], s[sgprWorkGroup1], s76 // wg1 += blockId * WGM


/* global read addresses: tile offset assignment a */

/* LVCA = 16 */
/* v0 = (local)groA-tile = serial/LVCA (note (wgA*MTA) will be added to SRD) */
/* v1 = groA-unroll = serial%LVCA */
v_lshrrev_b32 v0, 4, v[vgprSerial]                 // vectorStaticDiv: v0 = v[vgprSerial] / 16
v_and_b32 v1, 15, v[vgprSerial]                    // vectorStaticDiv: v1 = v[vgprSerial] % 16
/* gro-unroll *= glvw */
v_lshlrev_b32 v1, 1, v1                            // staticMultiply: v1 = v1 * 2
v_mov_b32 v4, v1                                   // copy for GlobalSplitU


/* global read addresses: tile offset assignment b */

/* LVCB = 16 */
/* v2 = (local)groB-tile = serial/LVCB (note (wgB*MTB) will be added to SRD) */
/* v3 = groB-unroll = serial%LVCB */
v_lshrrev_b32 v2, 4, v[vgprSerial]                 // vectorStaticDiv: v2 = v[vgprSerial] / 16
v_and_b32 v3, 15, v[vgprSerial]                    // vectorStaticDiv: v3 = v[vgprSerial] % 16
/* gro-unroll *= glvw */
v_lshlrev_b32 v3, 1, v3                            // staticMultiply: v3 = v3 * 2
v_mov_b32 v7, v3                                   // copy for GlobalSplitU


/* global read addresses: unroll assignment a */

/* v1 */


/* global read addresses: unroll assignment b */

/* v3 */


/* global read addresses: other summation assignments */

.set globalReadOffsetA0I,  0
.set globalReadOffsetB0I,  0


/* global read addresses: tile offsets a */



/* global read addresses: tile offsets b */



/* global read addresses: unroll offsets a */



/* global read addresses: unroll offsets b */



/* global read addresses: final offsets a */

GLOBAL_OFFSET_A vgprGlobalReadOffsetA+0,  1,  0, 8 // gROA_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetA+0], s[sgprStrideA0I], 16 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+0], s[sgprScalarGlobalReadOffsetA+0], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+1], s[sgprStrideA0I], 32 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+1], s[sgprScalarGlobalReadOffsetA+1], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetA+2], s[sgprStrideA0I], 48 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetA+2], s[sgprScalarGlobalReadOffsetA+2], 0x2 // scalar offset *= bytes/element


/* global read addresses: final offsets b */

GLOBAL_OFFSET_B vgprGlobalReadOffsetB+0,  3,  2, 8 // gROB_0_0_0_0
s_mul_i32 s[sgprScalarGlobalReadOffsetB+0], s[sgprStrideB1J], 16 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+0], s[sgprScalarGlobalReadOffsetB+0], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+1], s[sgprStrideB1J], 32 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+1], s[sgprScalarGlobalReadOffsetB+1], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+2], s[sgprStrideB1J], 48 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+2], s[sgprScalarGlobalReadOffsetB+2], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+3], s[sgprStrideB1J], 64 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+3], s[sgprScalarGlobalReadOffsetB+3], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+4], s[sgprStrideB1J], 80 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+4], s[sgprScalarGlobalReadOffsetB+4], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+5], s[sgprStrideB1J], 96 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+5], s[sgprScalarGlobalReadOffsetB+5], 0x2 // scalar offset *= bytes/element
s_mul_i32 s[sgprScalarGlobalReadOffsetB+6], s[sgprStrideB1J], 112 // compute offset diff (scaled tileDim)
s_lshl_b32 s[sgprScalarGlobalReadOffsetB+6], s[sgprScalarGlobalReadOffsetB+6], 0x2 // scalar offset *= bytes/element


/* global read addresses: addresses a */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s79, s[sgprWorkGroup0], 64            // WorkGroup[01] * MT
s_mul_i32 s78, s[sgprWorkGroup0], 64               // WorkGroup[01] * MT
s_mul_hi_u32 s79, s78, s[sgprStrideA0I]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s78, s78, s[sgprStrideA0I]               // tlu=0, scaled tile-offset by stride
s_mul_hi_u32 s77, 32, s[sgprGSUSumIdx]             // gsuOffset = DepthU*bpe*GSUSumIdx
s_mul_i32 s76, 32, s[sgprGSUSumIdx]                // gsuOffset = DepthU*bpe*GSUSumIdx
s_add_u32 s78, s78, s76                            // accum GsuOffet term to tilestart
s_addc_u32 s79, s79, s77                           // accum GsuOffet term to tilestart
s_sub_u32 s[sgprShadowLimitA+0], s[sgprTensor2dSizeA], s78 // sub tileStart
s_subb_u32 s[sgprShadowLimitA+1], s[sgprTensor2dSizeA+1], s79 // sub tileStart
s_lshl_b64 s[sgprShadowLimitA:sgprShadowLimitA+1], s[sgprShadowLimitA:sgprShadowLimitA+1], 0x2 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0], BufferLimit // Move shadow to real if we are within 2^32
s_lshl_b64 s[78:79], s[78:79], 2                   // tileStart *= BPE
s_add_u32 s[sgprSrdA+0], s[sgprAddressA+0], s78    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdA+1], s[sgprAddressA+1], s79   // SRD base = Address+ tileStart1
s_sub_u32 s[sgprSrdA+0], s[sgprSrdA+0], 8          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0         // pre-pad to make room for possible pointer shift
s_mov_b32 s[sgprSrdA+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: addresses b */

/* max read offset = size[n] * stride[n-1] */
s_mul_hi_u32 s79, s[sgprWorkGroup1], 128           // WorkGroup[01] * MT
s_mul_i32 s78, s[sgprWorkGroup1], 128              // WorkGroup[01] * MT
s_mul_hi_u32 s79, s78, s[sgprStrideB1J]            // tlu=0, scaled tile-offset by stride
s_mul_i32 s78, s78, s[sgprStrideB1J]               // tlu=0, scaled tile-offset by stride
s_mul_hi_u32 s77, 32, s[sgprGSUSumIdx]             // gsuOffset = DepthU*bpe*GSUSumIdx
s_mul_i32 s76, 32, s[sgprGSUSumIdx]                // gsuOffset = DepthU*bpe*GSUSumIdx
s_add_u32 s78, s78, s76                            // accum GsuOffet term to tilestart
s_addc_u32 s79, s79, s77                           // accum GsuOffet term to tilestart
s_sub_u32 s[sgprShadowLimitB+0], s[sgprTensor2dSizeB], s78 // sub tileStart
s_subb_u32 s[sgprShadowLimitB+1], s[sgprTensor2dSizeB+1], s79 // sub tileStart
s_lshl_b64 s[sgprShadowLimitB:sgprShadowLimitB+1], s[sgprShadowLimitB:sgprShadowLimitB+1], 0x2 // Set limit to use bytes
s_add_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], 8 // extend limit for pre-pad
s_addc_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // extend limit for pre-pad
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cselect_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0], BufferLimit // Move shadow to real if we are within 2^32
s_lshl_b64 s[78:79], s[78:79], 2                   // tileStart *= BPE
s_add_u32 s[sgprSrdB+0], s[sgprAddressB+0], s78    // SRD base = Address+ tileStart0
s_addc_u32 s[sgprSrdB+1], s[sgprAddressB+1], s79   // SRD base = Address+ tileStart1
s_sub_u32 s[sgprSrdB+0], s[sgprSrdB+0], 8          // pre-pad to make room for possible pointer shift
s_subb_u32 s[sgprSrdB+1], s[sgprSrdB+1], 0         // pre-pad to make room for possible pointer shift
s_mov_b32 s[sgprSrdB+3], Srd127_96                 // Set bits 127_96 in SRD


/* global read addresses: increments a */

s_mov_b32 s[sgprGlobalReadIncsA+1], DepthU*BpeA*3  // incrA (unrollIdx)


/* compute globalReadInc for higher-level loop */
s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+1], 5 // s[sgprLoopCounterL] = s[sgprSizesSum+1] / 32
s_mov_b32 s78, s[sgprLoopCounterL]                 // copy for divide IterGsu
s_mov_b32 s77, 0x0                                 // STATIC_DIV: divisior=3
s_mul_i32 s76, 0xaaaa, s78                         // tmp1 = dividend * magic hi
s_lshl_b64 s[76:77], s[76:77], 0x10                // left shift 16 bits
s_mul_i32 s[sgprLoopCounterL], s78, 0xaaab         // tmp0 = dividend * magic lo
s_add_u32 s76, s[sgprLoopCounterL], s76            // add lo
s_addc_u32 s77, s77, 0x0                           // add hi
s_lshr_b64 s[76:77], s[76:77], 0x21                // tmp1 = (dividend * magic) << shift
s_mov_b32 s[sgprLoopCounterL], s76                 // quotient
s_mul_i32 s76, s[sgprLoopCounterL], 0x3            // quotient*divisor
s_sub_u32 s[sgprGSUSumIdx+1], s78, s76             // rReg = dividend - quotient*divisor
s_add_u32 s76, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s76                // numIterMyWg++ if needed
s_mul_i32 s[sgprLoopCounterL], s[sgprLoopCounterL], 96 // =loopCounterName*DepthU
s_mul_i32 s[sgprGlobalReadIncsA+0], constStrideAL, s[sgprLoopCounterL] // tmp <- strideAL * myWgUnrollIters
s_sub_i32 s[sgprGlobalReadIncsA+0], s[sgprStrideAK], s[sgprGlobalReadIncsA+0] // incrAK = strideAK - <prev-incs>
s_lshl_b32 s[sgprGlobalReadIncsA+0], s[sgprGlobalReadIncsA+0], BpeALog2 // <- scale by bpe


/* global read addresses: increments b */

s_mov_b32 s[sgprGlobalReadIncsB+1], DepthU*BpeB*3  // incrB (unrollIdx)


/* compute globalReadInc for higher-level loop */
s_mul_i32 s[sgprGlobalReadIncsB+0], constStrideBL, s[sgprLoopCounterL] // tmp <- strideBL * myWgUnrollIters
s_sub_i32 s[sgprGlobalReadIncsB+0], s[sgprStrideBK], s[sgprGlobalReadIncsB+0] // incrBK = strideBK - <prev-incs>
s_lshl_b32 s[sgprGlobalReadIncsB+0], s[sgprGlobalReadIncsB+0], BpeBLog2 // <- scale by bpe


/******************************************/
/* Local Write Addresses                  */
/******************************************/

/* lwaTileAssignmentA = v0 */

/* lwaTileAssignmentB = v2 */

/* lwaUnrollAssignmentA = v4 */

/* lwaUnrollAssignmentB = v7 */


/* local write addresses: first offset a */

v_mul_u32_u24 v[vgprLocalWriteAddrA], 0x40, v4     // lwAL**(MTA + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrA], v0, v[vgprLocalWriteAddrA], 0x2 // lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe


/* local write addresses: first offset b */

v_mul_u32_u24 v[vgprLocalWriteAddrB], 0x80, v7     // lwBL**(MTB + PAD)
_v_add_lshl_u32 v[vgprLocalWriteAddrB], v2, v[vgprLocalWriteAddrB], 0x2 // lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpe
_v_add_co_u32 v[vgprLocalWriteAddrB], vcc, 0x2000, v[vgprLocalWriteAddrB] // lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=2048*4







/* declare loop num iterations */


v_accvgpr_write acc0, 0x0                          // init Acc vgprs
v_accvgpr_write acc1, 0x0                          // init Acc vgprs
v_accvgpr_write acc2, 0x0                          // init Acc vgprs
v_accvgpr_write acc3, 0x0                          // init Acc vgprs
v_accvgpr_write acc4, 0x0                          // init Acc vgprs
v_accvgpr_write acc5, 0x0                          // init Acc vgprs
v_accvgpr_write acc6, 0x0                          // init Acc vgprs
v_accvgpr_write acc7, 0x0                          // init Acc vgprs
v_accvgpr_write acc8, 0x0                          // init Acc vgprs
v_accvgpr_write acc9, 0x0                          // init Acc vgprs
v_accvgpr_write acc10, 0x0                         // init Acc vgprs
v_accvgpr_write acc11, 0x0                         // init Acc vgprs
v_accvgpr_write acc12, 0x0                         // init Acc vgprs
v_accvgpr_write acc13, 0x0                         // init Acc vgprs
v_accvgpr_write acc14, 0x0                         // init Acc vgprs
v_accvgpr_write acc15, 0x0                         // init Acc vgprs
v_accvgpr_write acc16, 0x0                         // init Acc vgprs
v_accvgpr_write acc17, 0x0                         // init Acc vgprs
v_accvgpr_write acc18, 0x0                         // init Acc vgprs
v_accvgpr_write acc19, 0x0                         // init Acc vgprs
v_accvgpr_write acc20, 0x0                         // init Acc vgprs
v_accvgpr_write acc21, 0x0                         // init Acc vgprs
v_accvgpr_write acc22, 0x0                         // init Acc vgprs
v_accvgpr_write acc23, 0x0                         // init Acc vgprs
v_accvgpr_write acc24, 0x0                         // init Acc vgprs
v_accvgpr_write acc25, 0x0                         // init Acc vgprs
v_accvgpr_write acc26, 0x0                         // init Acc vgprs
v_accvgpr_write acc27, 0x0                         // init Acc vgprs
v_accvgpr_write acc28, 0x0                         // init Acc vgprs
v_accvgpr_write acc29, 0x0                         // init Acc vgprs
v_accvgpr_write acc30, 0x0                         // init Acc vgprs
v_accvgpr_write acc31, 0x0                         // init Acc vgprs


/* summation loop 0 */


/* other summation, numIterK = sizeK */
s_mov_b32 s[sgprLoopCounterK], s[sgprSizesSum+0]   // init loop counter

openLoopK_8:
label_0009:

s_lshr_b32 s[sgprLoopCounterL], s[sgprSizesSum+1], 5 // s[sgprLoopCounterL] = s[sgprSizesSum+1] / 32
s_mov_b32 s78, s[sgprLoopCounterL]                 // copy for divide IterGsu
s_mov_b32 s77, 0x0                                 // STATIC_DIV: divisior=3
s_mul_i32 s76, 0xaaaa, s78                         // tmp1 = dividend * magic hi
s_lshl_b64 s[76:77], s[76:77], 0x10                // left shift 16 bits
s_mul_i32 s[sgprLoopCounterL], s78, 0xaaab         // tmp0 = dividend * magic lo
s_add_u32 s76, s[sgprLoopCounterL], s76            // add lo
s_addc_u32 s77, s77, 0x0                           // add hi
s_lshr_b64 s[76:77], s[76:77], 0x21                // tmp1 = (dividend * magic) << shift
s_mov_b32 s[sgprLoopCounterL], s76                 // quotient
s_mul_i32 s76, s[sgprLoopCounterL], 0x3            // quotient*divisor
s_sub_u32 s[sgprGSUSumIdx+1], s78, s76             // rReg = dividend - quotient*divisor
s_add_u32 s76, 1, s[sgprLoopCounterL]              // tmp<-numIterMyWg+
s_cmp_lt_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx < numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], s76                // numIterMyWg++ if needed
s_mov_b32 s[sgprOrigLoopCounter], s[sgprLoopCounterL] // copy loop counter

/* local read addresses: init pointers a */


/* localReadInitPointers */

/* local read addresses: init pointers b */


/* localReadInitPointers */


/******************************************/
/* End setupNewTile                       */
/******************************************/


/******************************************/
/* Unrolled Loop(s) - Begin               */
/******************************************/

openLoopL_11:
s_cmp_le_u32 s[sgprLoopCounterL], 0x0              // LoopCounterL < EndCounter
s_cbranch_scc1 label_0002                          // don't enter LoopL
label_0001:


/******************************************/
/* Unroll Loop 1/1 - Begin                */
/******************************************/

label_0012: // LoopCopy1 

buffer_load_dwordx2 v[vgprG2LA+0:vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx2 v[vgprG2LA+2:vgprG2LA+2+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0], offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx2 v[vgprG2LA+4:vgprG2LA+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1], offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx2 v[vgprG2LA+6:vgprG2LA+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2], offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx2 v[vgprG2LB+0:vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // G -> Reg 0_0_0_0
buffer_load_dwordx2 v[vgprG2LB+2:vgprG2LB+2+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // G -> Reg 0_0_1_0
buffer_load_dwordx2 v[vgprG2LB+4:vgprG2LB+4+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1], offen offset:0 // G -> Reg 0_0_2_0
buffer_load_dwordx2 v[vgprG2LB+6:vgprG2LB+6+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2], offen offset:0 // G -> Reg 0_0_3_0
buffer_load_dwordx2 v[vgprG2LB+8:vgprG2LB+8+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3], offen offset:0 // G -> Reg 0_0_4_0
buffer_load_dwordx2 v[vgprG2LB+10:vgprG2LB+10+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4], offen offset:0 // G -> Reg 0_0_5_0
buffer_load_dwordx2 v[vgprG2LB+12:vgprG2LB+12+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+5], offen offset:0 // G -> Reg 0_0_6_0
buffer_load_dwordx2 v[vgprG2LB+14:vgprG2LB+14+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+6], offen offset:0 // G -> Reg 0_0_7_0

/* global read inc A loopL */
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+1] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+1] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopL */
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+1] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], 0        // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+1] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], 0 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32

s_waitcnt vmcnt(0)                                 // 5wait for global read

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //PGR=0, prior iter done reading lds


/* local write a */

ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:256 // lwoA_0_1_0_0 = (1 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 256
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:64 // lwoA_0_0_1_0 = (0 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 64
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:320 // lwoA_0_1_1_0 = (1 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 320
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+4] offset:128 // lwoA_0_0_2_0 = (0 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 128
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+5] offset:384 // lwoA_0_1_2_0 = (1 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 384
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+6] offset:192 // lwoA_0_0_3_0 = (0 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 192
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+7] offset:448 // lwoA_0_1_3_0 = (1 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 448


/* local write b */

ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:512 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 512
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:64 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 64
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:576 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 576
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:128 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 128
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:640 // lwoB_0_1_2_0 = (1 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 640
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:192 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 192
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:704 // lwoB_0_1_3_0 = (1 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 704
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+8] offset:256 // lwoB_0_0_4_0 = (0 + 0*LSCB)*(MT1J+PAD) + (4*LSPB) = 256
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+9] offset:768 // lwoB_0_1_4_0 = (1 + 0*LSCB)*(MT1J+PAD) + (4*LSPB) = 768
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+10] offset:320 // lwoB_0_0_5_0 = (0 + 0*LSCB)*(MT1J+PAD) + (5*LSPB) = 320
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+11] offset:832 // lwoB_0_1_5_0 = (1 + 0*LSCB)*(MT1J+PAD) + (5*LSPB) = 832
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+12] offset:384 // lwoB_0_0_6_0 = (0 + 0*LSCB)*(MT1J+PAD) + (6*LSPB) = 384
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+13] offset:896 // lwoB_0_1_6_0 = (1 + 0*LSCB)*(MT1J+PAD) + (6*LSPB) = 896
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+14] offset:448 // lwoB_0_0_7_0 = (0 + 0*LSCB)*(MT1J+PAD) + (7*LSPB) = 448
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+15] offset:960 // lwoB_0_1_7_0 = (1 + 0*LSCB)*(MT1J+PAD) + (7*LSPB) = 960

s_waitcnt lgkmcnt(0)                               // 2prefetch wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //




/* iter 0 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->64 */

/* local read increment b */
/* N/A, lro->128 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 1 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:256 // L -> Reg lro=64 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:512 // L -> Reg lro=128 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->128 */

/* local read increment b */
/* N/A, lro->256 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 2 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:512 // L -> Reg lro=128 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:1024 // L -> Reg lro=256 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->192 */

/* local read increment b */
/* N/A, lro->384 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 3 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:768 // L -> Reg lro=192 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:1536 // L -> Reg lro=384 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->256 */

/* local read increment b */
/* N/A, lro->512 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 4 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:1024 // L -> Reg lro=256 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:2048 // L -> Reg lro=512 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->320 */

/* local read increment b */
/* N/A, lro->640 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 5 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:1280 // L -> Reg lro=320 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:2560 // L -> Reg lro=640 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->384 */

/* local read increment b */
/* N/A, lro->768 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 6 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:1536 // L -> Reg lro=384 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:3072 // L -> Reg lro=768 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->448 */

/* local read increment b */
/* N/A, lro->896 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 7 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:1792 // L -> Reg lro=448 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:3584 // L -> Reg lro=896 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->512 */

/* local read increment b */
/* N/A, lro->1024 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 8 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:2048 // L -> Reg lro=512 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:4096 // L -> Reg lro=1024 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->576 */

/* local read increment b */
/* N/A, lro->1152 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 9 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:2304 // L -> Reg lro=576 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:4608 // L -> Reg lro=1152 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->640 */

/* local read increment b */
/* N/A, lro->1280 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 10 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:2560 // L -> Reg lro=640 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:5120 // L -> Reg lro=1280 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->704 */

/* local read increment b */
/* N/A, lro->1408 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 11 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:2816 // L -> Reg lro=704 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:5632 // L -> Reg lro=1408 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->768 */

/* local read increment b */
/* N/A, lro->1536 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 12 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:3072 // L -> Reg lro=768 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:6144 // L -> Reg lro=1536 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->832 */

/* local read increment b */
/* N/A, lro->1664 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 13 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:3328 // L -> Reg lro=832 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:6656 // L -> Reg lro=1664 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->896 */

/* local read increment b */
/* N/A, lro->1792 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 14 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:3584 // L -> Reg lro=896 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:7168 // L -> Reg lro=1792 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->960 */

/* local read increment b */
/* N/A, lro->1920 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 15 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:3840 // L -> Reg lro=960 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:7680 // L -> Reg lro=1920 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1024 */

/* local read increment b */
/* N/A, lro->2048 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 16 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:4096 // L -> Reg lro=1024 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:8192 // L -> Reg lro=2048 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1088 */

/* local read increment b */
/* N/A, lro->2176 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 17 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:4352 // L -> Reg lro=1088 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:8704 // L -> Reg lro=2176 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1152 */

/* local read increment b */
/* N/A, lro->2304 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 18 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:4608 // L -> Reg lro=1152 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:9216 // L -> Reg lro=2304 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1216 */

/* local read increment b */
/* N/A, lro->2432 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 19 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:4864 // L -> Reg lro=1216 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:9728 // L -> Reg lro=2432 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1280 */

/* local read increment b */
/* N/A, lro->2560 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 20 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:5120 // L -> Reg lro=1280 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:10240 // L -> Reg lro=2560 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1344 */

/* local read increment b */
/* N/A, lro->2688 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 21 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:5376 // L -> Reg lro=1344 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:10752 // L -> Reg lro=2688 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1408 */

/* local read increment b */
/* N/A, lro->2816 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 22 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:5632 // L -> Reg lro=1408 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:11264 // L -> Reg lro=2816 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1472 */

/* local read increment b */
/* N/A, lro->2944 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 23 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:5888 // L -> Reg lro=1472 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:11776 // L -> Reg lro=2944 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1536 */

/* local read increment b */
/* N/A, lro->3072 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 24 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:6144 // L -> Reg lro=1536 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:12288 // L -> Reg lro=3072 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1600 */

/* local read increment b */
/* N/A, lro->3200 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 25 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:6400 // L -> Reg lro=1600 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:12800 // L -> Reg lro=3200 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1664 */

/* local read increment b */
/* N/A, lro->3328 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 26 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:6656 // L -> Reg lro=1664 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:13312 // L -> Reg lro=3328 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1728 */

/* local read increment b */
/* N/A, lro->3456 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 27 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:6912 // L -> Reg lro=1728 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:13824 // L -> Reg lro=3456 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1792 */

/* local read increment b */
/* N/A, lro->3584 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 28 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:7168 // L -> Reg lro=1792 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:14336 // L -> Reg lro=3584 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1856 */

/* local read increment b */
/* N/A, lro->3712 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 29 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:7424 // L -> Reg lro=1856 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:14848 // L -> Reg lro=3712 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1920 */

/* local read increment b */
/* N/A, lro->3840 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* iter 30 */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:7680 // L -> Reg lro=1920 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:15360 // L -> Reg lro=3840 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read increment a */
/* N/A, lro->1984 */

/* local read increment b */
/* N/A, lro->3968 */
s_waitcnt lgkmcnt(0)                               // wait for prior local read local write old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]




/* iter 31 (last) */


/* local read a */
ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:7936 // L -> Reg lro=1984 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read b */
ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:15872 // L -> Reg lro=3968 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0

/* local read init pointers a */

/* localReadInitPointers */

/* local read init pointers b */

/* localReadInitPointers */
s_waitcnt lgkmcnt(0)                               // 1wait for local read old=0 new=0 (Local write no wait)
v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/******************************************/
/* Unrolled Loop - End                    */
/******************************************/


/* closeLoop loopL finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1 // dec counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==0
s_cbranch_scc0 label_0001                          // restart LoopL
label_0003: // unroll loop odditer exit
label_0002:


/******************************************/
/* Tail Loop                              */
/******************************************/


//numIterL = (((sizeL % LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU)
s_and_b32 s[sgprLoopCounterL], 31, s[sgprSizesSum+1] // s[sgprLoopCounterL] = s[sgprSizesSum+1] % 32
/* calculate number of remaining loops in terms of how many matrix instructions */
//numIterL = ((numIterL + MatrixInstL - 1) / MatrixInstL)
s_add_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0 // 
s_lshr_b32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0 // s[sgprLoopCounterL] = s[sgprLoopCounterL] / 1
s_cmp_lg_u32 s[sgprGSUSumIdx], s[sgprGSUSumIdx+1]  // gsuSumIdx == numIterPerWgRemainder
s_cmov_b32 s[sgprLoopCounterL], 0x0                // numIter=0 if gsuSimIdx!=remainder
s_cmp_eq_u32 s[sgprLoopCounterL], 0x0              // numIterL == 0
s_cbranch_scc1 label_0006                          // skip to end of tail loop b/c numIter==0


/* global read a */

/* g2l=0, load component 0 */
buffer_load_dword v[vgprG2LA+0+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_dword v[vgprG2LA+0+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], 0, offen offset:4 // load one buffer value
/* g2l=2, load component 0 */
buffer_load_dword v[vgprG2LA+2+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0], offen offset:0 // load one buffer value
/* g2l=2, load component 1 */
buffer_load_dword v[vgprG2LA+2+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+0], offen offset:4 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_dword v[vgprG2LA+4+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1], offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_dword v[vgprG2LA+4+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+1], offen offset:4 // load one buffer value
/* g2l=6, load component 0 */
buffer_load_dword v[vgprG2LA+6+0], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2], offen offset:0 // load one buffer value
/* g2l=6, load component 1 */
buffer_load_dword v[vgprG2LA+6+1], v[vgprGlobalReadOffsetA+0], s[sgprSrdA:sgprSrdA+3], s[sgprScalarGlobalReadOffsetA+2], offen offset:4 // load one buffer value


/* global read b */

/* g2l=0, load component 0 */
buffer_load_dword v[vgprG2LB+0+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:0 // load one buffer value
/* g2l=0, load component 1 */
buffer_load_dword v[vgprG2LB+0+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], 0, offen offset:4 // load one buffer value
/* g2l=2, load component 0 */
buffer_load_dword v[vgprG2LB+2+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:0 // load one buffer value
/* g2l=2, load component 1 */
buffer_load_dword v[vgprG2LB+2+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+0], offen offset:4 // load one buffer value
/* g2l=4, load component 0 */
buffer_load_dword v[vgprG2LB+4+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1], offen offset:0 // load one buffer value
/* g2l=4, load component 1 */
buffer_load_dword v[vgprG2LB+4+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+1], offen offset:4 // load one buffer value
/* g2l=6, load component 0 */
buffer_load_dword v[vgprG2LB+6+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2], offen offset:0 // load one buffer value
/* g2l=6, load component 1 */
buffer_load_dword v[vgprG2LB+6+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+2], offen offset:4 // load one buffer value
/* g2l=8, load component 0 */
buffer_load_dword v[vgprG2LB+8+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3], offen offset:0 // load one buffer value
/* g2l=8, load component 1 */
buffer_load_dword v[vgprG2LB+8+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+3], offen offset:4 // load one buffer value
/* g2l=10, load component 0 */
buffer_load_dword v[vgprG2LB+10+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4], offen offset:0 // load one buffer value
/* g2l=10, load component 1 */
buffer_load_dword v[vgprG2LB+10+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+4], offen offset:4 // load one buffer value
/* g2l=12, load component 0 */
buffer_load_dword v[vgprG2LB+12+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+5], offen offset:0 // load one buffer value
/* g2l=12, load component 1 */
buffer_load_dword v[vgprG2LB+12+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+5], offen offset:4 // load one buffer value
/* g2l=14, load component 0 */
buffer_load_dword v[vgprG2LB+14+0], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+6], offen offset:0 // load one buffer value
/* g2l=14, load component 1 */
buffer_load_dword v[vgprG2LB+14+1], v[vgprGlobalReadOffsetB+0], s[sgprSrdB:sgprSrdB+3], s[sgprScalarGlobalReadOffsetB+6], offen offset:4 // load one buffer value

s_waitcnt vmcnt(0)                                 // 2wait for global read

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //




/* local write a */

ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+0] offset:0 // lwoA_0_0_0_0 = (0 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 0
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+1] offset:256 // lwoA_0_1_0_0 = (1 + 0*LSCA)*(MT0I+PAD) + (0*LSPA) = 256
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+2] offset:64 // lwoA_0_0_1_0 = (0 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 64
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+3] offset:320 // lwoA_0_1_1_0 = (1 + 0*LSCA)*(MT0I+PAD) + (1*LSPA) = 320
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+4] offset:128 // lwoA_0_0_2_0 = (0 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 128
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+5] offset:384 // lwoA_0_1_2_0 = (1 + 0*LSCA)*(MT0I+PAD) + (2*LSPA) = 384
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+6] offset:192 // lwoA_0_0_3_0 = (0 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 192
ds_write_b32 v[vgprLocalWriteAddrA], v[vgprG2LA+7] offset:448 // lwoA_0_1_3_0 = (1 + 0*LSCA)*(MT0I+PAD) + (3*LSPA) = 448


/* local write b */

ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+0] offset:0 // lwoB_0_0_0_0 = (0 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 0
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+1] offset:512 // lwoB_0_1_0_0 = (1 + 0*LSCB)*(MT1J+PAD) + (0*LSPB) = 512
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+2] offset:64 // lwoB_0_0_1_0 = (0 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 64
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+3] offset:576 // lwoB_0_1_1_0 = (1 + 0*LSCB)*(MT1J+PAD) + (1*LSPB) = 576
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+4] offset:128 // lwoB_0_0_2_0 = (0 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 128
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+5] offset:640 // lwoB_0_1_2_0 = (1 + 0*LSCB)*(MT1J+PAD) + (2*LSPB) = 640
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+6] offset:192 // lwoB_0_0_3_0 = (0 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 192
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+7] offset:704 // lwoB_0_1_3_0 = (1 + 0*LSCB)*(MT1J+PAD) + (3*LSPB) = 704
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+8] offset:256 // lwoB_0_0_4_0 = (0 + 0*LSCB)*(MT1J+PAD) + (4*LSPB) = 256
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+9] offset:768 // lwoB_0_1_4_0 = (1 + 0*LSCB)*(MT1J+PAD) + (4*LSPB) = 768
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+10] offset:320 // lwoB_0_0_5_0 = (0 + 0*LSCB)*(MT1J+PAD) + (5*LSPB) = 320
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+11] offset:832 // lwoB_0_1_5_0 = (1 + 0*LSCB)*(MT1J+PAD) + (5*LSPB) = 832
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+12] offset:384 // lwoB_0_0_6_0 = (0 + 0*LSCB)*(MT1J+PAD) + (6*LSPB) = 384
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+13] offset:896 // lwoB_0_1_6_0 = (1 + 0*LSCB)*(MT1J+PAD) + (6*LSPB) = 896
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+14] offset:448 // lwoB_0_0_7_0 = (0 + 0*LSCB)*(MT1J+PAD) + (7*LSPB) = 448
ds_write_b32 v[vgprLocalWriteAddrB], v[vgprG2LB+15] offset:960 // lwoB_0_1_7_0 = (1 + 0*LSCB)*(MT1J+PAD) + (7*LSPB) = 960

s_waitcnt lgkmcnt(0)                               // 5wait for local write

s_waitcnt lgkmcnt(0) & vmcnt(0)                    // force waitcnt0
s_barrier //


/* tail loop: macs */

s_cmp_le_u32 s[sgprLoopCounterL], 0x0              // LoopCounterL < EndCounter
s_cbranch_scc1 label_0006                          // don't enter LoopL
s_mov_b32 s[sgprOrigLoopCounter], 0                // repurpose to count each localRead increment
label_0005:


/* local read a */

ds_read_b32 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read b */

ds_read_b32 v[vgprValuB_X0_I0+0], v[vgprLocalReadAddrB] offset:0 // L -> Reg lro=0 swapByteOffset=0 ti=4 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0


/* local read inc a */

s_mov_b32 s75, 0x100                               // inc
_v_add_co_u32 v[vgprLocalReadAddrA], vcc, s75, v[vgprLocalReadAddrA] // lrA += 256 (LSU*(MT+PAD)*bpe)


/* local read inc b */

s_mov_b32 s75, 0x200                               // inc
_v_add_co_u32 v[vgprLocalReadAddrB], vcc, s75, v[vgprLocalReadAddrB] // lrB += 512 (LSU*(MT+PAD)*bpe)

s_waitcnt lgkmcnt(0)                               // 4wait for local read

v_mfma_f32_32x32x1f32 a[0:31], v[vgprValuA_X0_I0+0], v[vgprValuB_X0_I0+0], a[0:31]


/* closeLoop loopL finalLoop=1 tailLoop=1 */
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 0x1 // dec counterL (tailLoop)
s_add_u32 s[sgprOrigLoopCounter], s[sgprOrigLoopCounter], 0x1 // inc counterL
s_cmp_eq_i32 s[sgprLoopCounterL], 0x0              // counterL==0
s_cbranch_scc0 label_0005                          // restart LoopL
s_mov_b32 s75, 256                                 // tailloop lds offset
s_mul_i32 s75, s[sgprOrigLoopCounter], s75         // scale by mul
v_sub_u32 v[vgprLocalReadAddrA], v[vgprLocalReadAddrA], s75 // remove lro damage
s_mov_b32 s75, 512                                 // tailloop lds offset
s_mul_i32 s75, s[sgprOrigLoopCounter], s75         // scale by mul
v_sub_u32 v[vgprLocalReadAddrB], v[vgprLocalReadAddrB], s75 // remove lro damage
label_0006:


/* global read inc AB */


/* global read inc A loopK */
s_ashr_i32 s75, s[sgprGlobalReadIncsA+0], 31       // sign-extend
s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdA+1], s[sgprSrdA+1], s75      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitA+0], s[sgprShadowLimitA+0], s[sgprGlobalReadIncsA+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitA+1], s[sgprShadowLimitA+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitA+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdA+2], s[sgprShadowLimitA+0]    // Move shadow to real if we are within 2^32

/* global read inc B loopK */
s_ashr_i32 s75, s[sgprGlobalReadIncsB+0], 31       // sign-extend
s_add_u32 s[sgprSrdB+0], s[sgprSrdB+0], s[sgprGlobalReadIncsB+0] // gra SRD += inc(lower)
s_addc_u32  s[sgprSrdB+1], s[sgprSrdB+1], s75      // gra SRD += inc(upper)
s_sub_u32 s[sgprShadowLimitB+0], s[sgprShadowLimitB+0], s[sgprGlobalReadIncsB+0] // limit -= inc)
s_subb_u32 s[sgprShadowLimitB+1], s[sgprShadowLimitB+1], s75 // limit -= inc)
s_cmp_eq_u32 s[sgprShadowLimitB+1], 0              // are we within 2^32?
s_cmov_b32 s[sgprSrdB+2], s[sgprShadowLimitB+0]    // Move shadow to real if we are within 2^32


/* closeLoop loopK finalLoop=1 tailLoop=0 */
s_sub_u32 s[sgprLoopCounterK], s[sgprLoopCounterK], 1 // dec counterK
s_cmp_eq_i32 s[sgprLoopCounterK], 0x0              // counterK==0
s_cbranch_scc0 label_0009                          // restart LoopK
label_0010:

Summation_End_15:
/* endSummation: add vgpr 32...60 to pool */
s_nop 16

/* Mapping of Acc register -> C Vgpr register */
v_accvgpr_read_b32 v[vgprValuC+0], acc0            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+1], acc1            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+2], acc2            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+3], acc3            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+4], acc4            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+5], acc5            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+6], acc6            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+7], acc7            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+8], acc8            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+9], acc9            // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+10], acc10          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+11], acc11          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+12], acc12          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+13], acc13          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+14], acc14          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+15], acc15          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+16], acc16          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+17], acc17          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+18], acc18          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+19], acc19          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+20], acc20          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+21], acc21          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+22], acc22          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+23], acc23          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+24], acc24          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+25], acc25          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+26], acc26          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+27], acc27          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+28], acc28          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+29], acc29          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+30], acc30          // copy areg to vreg
v_accvgpr_read_b32 v[vgprValuC+31], acc31          // copy areg to vreg

s_mov_b32 s[sgprSrdD+0], s[sgprAddressD+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdD+1], s[sgprAddressD+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdD+2], 0x80000000                // 
s_mov_b32 s[sgprSrdD+3], Srd127_96                 // Set bits 127_96 in post-loop SRD

s_mov_b32 s[sgprSrdC+0], s[sgprAddressC+0]         // init SRD base address (lower)
s_mov_b32 s[sgprSrdC+1], s[sgprAddressC+1]         // init SRD base address (upper) + other fields
s_mov_b32 s[sgprSrdC+2], 0x80000000                // 
s_mov_b32 s[sgprSrdC+3], Srd127_96                 // Set bits 127_96 in post-loop SRD


s_mul_i32 s56, 0x80, s[sgprWorkGroup1]             // <- wg1*MT1
s_mul_hi_u32 s55, s56, s[sgprStrideC1J]            // CScale s56 by Stride
s_mul_i32 s54, s56, s[sgprStrideC1J]               // CScale s56 by Stride
s_lshl_b64 s[54:55], s[54:55], 2                   // scale by bpe
s_add_u32 s[sgprSrdC+0], s[sgprSrdC+0], s54        // add lo to SRD
s_addc_u32 s[sgprSrdC+1], s[sgprSrdC+1], s55       // add hi to SRD
s_add_u32 s[sgprSrdD+0], s[sgprSrdD+0], s54        // add lo to SRD
s_addc_u32 s[sgprSrdD+1], s[sgprSrdD+1], s55       // add hi to SRD




/* not-LocalSplitU: global write indices */

/* computeStoreVgprs */
v_lshrrev_b32 v33, 6, v[vgprSerial]                // vectorStaticDiv: v33 = v[vgprSerial] / 64
v_and_b32 v32, 63, v[vgprSerial]                   // vectorStaticDiv: v32 = v[vgprSerial] % 64
v_mul_lo_u32 v33, 0x20, v33                        // col element offset for each block
v_mul_lo_u32 v34, v33, s[sgprStrideC1J]            // Col-block-offset = Col-id*Stride
v_and_b32 v36, 0x1f, v[vgprSerial]                 // colId-perBlock= vgprSerial%MatrixInstN
v_mul_lo_u32 v37, v36, s[sgprStrideC1J]            // 
v_add_u32 v34, v37, v34                            // rowStart VGPR
v_add_u32 v33, v36, v33                            // coord1 offset in MacroTile

v_lshrrev_b32 v38, 0x5, v32                        // vectorStaticDiv vgprTmp = tid0 / matrixInstM
v_lshlrev_b32 v32, 0x2, v38                        // tmpV3 = tmpV3 << 2 (4xMatrixInstN per block

s_mul_i32 s54, 0x40, s[sgprWorkGroup0]             // wgp0 * MT0
v_add_co_u32 v32, vcc, s54, v32                    // coord0 = (tid0 / matrixInstM)<<2 + wg0*MT0
s_mul_i32 s56, 0x80, s[sgprWorkGroup1]             // <- wg1*MT1
_v_add_co_u32 v33, vcc, s56, v33                   // coord1 = tid1*VW + wg1*MT1


/* not-LocalSplitU: global write */

s_and_b32 s54, 63, s[sgprSizeI]                    // s54 = s[sgprSizeI] % 64
s_add_u32 s56, -0x1, s[sgprNumWorkGroups0]         // 
s_cmp_ge_u32 s[sgprWorkGroup0], s56                // wg0 >= nwg0-1 ?
s_cselect_b32 s54, s54, 0                          // set rMT0
s_cmpk_gt_u32 s54, 0x0                             // rMT0 > 0
s_cbranch_scc1 GW_B0_E1_21                         // jump if edges required
s_and_b32 s54, 127, s[sgprSizeJ]                   // s54 = s[sgprSizeJ] % 128
s_add_u32 s56, -0x1, s[sgprNumWorkGroups1]         // 
s_cmp_ge_u32 s[sgprWorkGroup1], s56                // wg1 >= nwg1-1
s_cselect_b32 s54, s54, 0                          // set rMT1
s_cmpk_gt_u32 s54, 0x0                             // rMT1 > 0
s_cbranch_scc1 GW_B0_E1_21                         // jump if edges required
GW_B0_E0_18:

/* allocate 20 sgpr. perBatch=6 perElement=2 elements=7 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1:vaw:1); (0,0,0,1:vw1:vaw:1); (0,0,0,2:vw1:vaw:1); (0,0,0,3:vw1:vaw:1); (0,1,0,0:vw1:vaw:1); (0,1,0,1:vw1:vaw:1); (0,1,0,2:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
_v_add_lshl_u32 v37, v34, v32, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v35, vcc, v32, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
_v_add_co_u32 v35, vcc, v32, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
_v_add_co_u32 v35, vcc, v32, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v35, vcc, v32, 8                     // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
_v_add_co_u32 v35, vcc, v32, 9                     // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
_v_add_co_u32 v35, vcc, v32, 10                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+0]                 // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+1]                 // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+2]                 // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+3]                 // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+4]                 // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+5]                 // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+6]                 // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0025                         // if exec is zero skip loop

/* atomic CAS loop */
label_0024:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+0]                 // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+1]                 // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+2]                 // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+3]                 // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+4]                 // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+5]                 // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+6]                 // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0024                        // try again if not complete
label_0025:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,1,0,3:vw1:vaw:1); (0,2,0,0:vw1:vaw:1); (0,2,0,1:vw1:vaw:1); (0,2,0,2:vw1:vaw:1); (0,2,0,3:vw1:vaw:1); (0,3,0,0:vw1:vaw:1); (0,3,0,1:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
_v_add_co_u32 v35, vcc, v32, 11                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v35, vcc, v32, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,2,1) */
_v_add_co_u32 v35, vcc, v32, 17                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,2,2) */
_v_add_co_u32 v35, vcc, v32, 18                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,2,3) */
_v_add_co_u32 v35, vcc, v32, 19                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_v_add_co_u32 v35, vcc, v32, 24                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,3,1) */
_v_add_co_u32 v35, vcc, v32, 25                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 1, 0, 3), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 0, 2), (0, 2, 0, 3), (0, 3, 0, 0), (0, 3, 0, 1)] */
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+7]                 // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+8]                 // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+9]                 // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+10]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+11]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+12]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+13]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0027                         // if exec is zero skip loop

/* atomic CAS loop */
label_0026:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+7]                 // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+8]                 // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+9]                 // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+10]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+11]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+12]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+13]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0026                        // try again if not complete
label_0027:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,3,0,2:vw1:vaw:1); (0,3,0,3:vw1:vaw:1); (0,4,0,0:vw1:vaw:1); (0,4,0,1:vw1:vaw:1); (0,4,0,2:vw1:vaw:1); (0,4,0,3:vw1:vaw:1); (0,5,0,0:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,3,2) */
_v_add_co_u32 v35, vcc, v32, 26                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,3,3) */
_v_add_co_u32 v35, vcc, v32, 27                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
_v_add_co_u32 v35, vcc, v32, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,4,1) */
_v_add_co_u32 v35, vcc, v32, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,4,2) */
_v_add_co_u32 v35, vcc, v32, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,4,3) */
_v_add_co_u32 v35, vcc, v32, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
_v_add_co_u32 v35, vcc, v32, 40                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 3, 0, 2), (0, 3, 0, 3), (0, 4, 0, 0), (0, 4, 0, 1), (0, 4, 0, 2), (0, 4, 0, 3), (0, 5, 0, 0)] */
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+14]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+15]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+16]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+17]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+18]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+19]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+20]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0029                         // if exec is zero skip loop

/* atomic CAS loop */
label_0028:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+14]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+15]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+16]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+17]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+18]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+19]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+20]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0028                        // try again if not complete
label_0029:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,5,0,1:vw1:vaw:1); (0,5,0,2:vw1:vaw:1); (0,5,0,3:vw1:vaw:1); (0,6,0,0:vw1:vaw:1); (0,6,0,1:vw1:vaw:1); (0,6,0,2:vw1:vaw:1); (0,6,0,3:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,5,1) */
_v_add_co_u32 v35, vcc, v32, 41                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,5,2) */
_v_add_co_u32 v35, vcc, v32, 42                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,5,3) */
_v_add_co_u32 v35, vcc, v32, 43                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
_v_add_co_u32 v35, vcc, v32, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,6,1) */
_v_add_co_u32 v35, vcc, v32, 49                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,6,2) */
_v_add_co_u32 v35, vcc, v32, 50                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,6,3) */
_v_add_co_u32 v35, vcc, v32, 51                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 5, 0, 1), (0, 5, 0, 2), (0, 5, 0, 3), (0, 6, 0, 0), (0, 6, 0, 1), (0, 6, 0, 2), (0, 6, 0, 3)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+21]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+22]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+23]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+24]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+25]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+26]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+27]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0031                         // if exec is zero skip loop

/* atomic CAS loop */
label_0030:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+21]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+22]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+23]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+24]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+25]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+26]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+27]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0030                        // try again if not complete
label_0031:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #4 (d1,d0,vc1,vc0) = */
/*    (0,7,0,0:vw1:vaw:1); (0,7,0,1:vw1:vaw:1); (0,7,0,2:vw1:vaw:1); (0,7,0,3:vw1:vaw:1); (0,8,0,0:vw1:vaw:1); (0,8,0,1:vw1:vaw:1); (0,8,0,2:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
_v_add_co_u32 v35, vcc, v32, 56                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,7,1) */
_v_add_co_u32 v35, vcc, v32, 57                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,7,2) */
_v_add_co_u32 v35, vcc, v32, 58                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,7,3) */
_v_add_co_u32 v35, vcc, v32, 59                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,8,0) */
_v_add_co_u32 v35, vcc, v32, 64                    // coord0.1: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,8,1) */
s_mov_b32 s54, 65                                  // coordOffset0 d0=8 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,8,2) */
s_mov_b32 s54, 66                                  // coordOffset0 d0=8 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 7, 0, 0), (0, 7, 0, 1), (0, 7, 0, 2), (0, 7, 0, 3), (0, 8, 0, 0), (0, 8, 0, 1), (0, 8, 0, 2)] */
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
v_mul_f32 v[vgprValuC+32], s[sgprAlpha], v[vgprValuC+32] // *= alpha
v_mul_f32 v[vgprValuC+33], s[sgprAlpha], v[vgprValuC+33] // *= alpha
v_mul_f32 v[vgprValuC+34], s[sgprAlpha], v[vgprValuC+34] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+28]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+29]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+30]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+31]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+32]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+33]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+34]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0033                         // if exec is zero skip loop

/* atomic CAS loop */
label_0032:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+28]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+29]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+30]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+31]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+32]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+33]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+34]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0032                        // try again if not complete
label_0033:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #5 (d1,d0,vc1,vc0) = */
/*    (0,8,0,3:vw1:vaw:1); (0,9,0,0:vw1:vaw:1); (0,9,0,1:vw1:vaw:1); (0,9,0,2:vw1:vaw:1); (0,9,0,3:vw1:vaw:1); (0,10,0,0:vw1:vaw:1); (0,10,0,1:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,8,3) */
s_mov_b32 s54, 67                                  // coordOffset0 d0=8 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,9,0) */
s_mov_b32 s54, 72                                  // coordOffset0 d0=9 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,9,1) */
s_mov_b32 s54, 73                                  // coordOffset0 d0=9 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,9,2) */
s_mov_b32 s54, 74                                  // coordOffset0 d0=9 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,9,3) */
s_mov_b32 s54, 75                                  // coordOffset0 d0=9 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,10,0) */
s_mov_b32 s54, 80                                  // coordOffset0 d0=10 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,10,1) */
s_mov_b32 s54, 81                                  // coordOffset0 d0=10 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 8, 0, 3), (0, 9, 0, 0), (0, 9, 0, 1), (0, 9, 0, 2), (0, 9, 0, 3), (0, 10, 0, 0), (0, 10, 0, 1)] */
v_mul_f32 v[vgprValuC+35], s[sgprAlpha], v[vgprValuC+35] // *= alpha
v_mul_f32 v[vgprValuC+36], s[sgprAlpha], v[vgprValuC+36] // *= alpha
v_mul_f32 v[vgprValuC+37], s[sgprAlpha], v[vgprValuC+37] // *= alpha
v_mul_f32 v[vgprValuC+38], s[sgprAlpha], v[vgprValuC+38] // *= alpha
v_mul_f32 v[vgprValuC+39], s[sgprAlpha], v[vgprValuC+39] // *= alpha
v_mul_f32 v[vgprValuC+40], s[sgprAlpha], v[vgprValuC+40] // *= alpha
v_mul_f32 v[vgprValuC+41], s[sgprAlpha], v[vgprValuC+41] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+35]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+36]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+37]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+38]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+39]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+40]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+41]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0035                         // if exec is zero skip loop

/* atomic CAS loop */
label_0034:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+35]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+36]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+37]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+38]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+39]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+40]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+41]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0034                        // try again if not complete
label_0035:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #6 (d1,d0,vc1,vc0) = */
/*    (0,10,0,2:vw1:vaw:1); (0,10,0,3:vw1:vaw:1); (0,11,0,0:vw1:vaw:1); (0,11,0,1:vw1:vaw:1); (0,11,0,2:vw1:vaw:1); (0,11,0,3:vw1:vaw:1); (0,12,0,0:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,10,2) */
s_mov_b32 s54, 82                                  // coordOffset0 d0=10 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,10,3) */
s_mov_b32 s54, 83                                  // coordOffset0 d0=10 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,11,0) */
s_mov_b32 s54, 88                                  // coordOffset0 d0=11 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,11,1) */
s_mov_b32 s54, 89                                  // coordOffset0 d0=11 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,11,2) */
s_mov_b32 s54, 90                                  // coordOffset0 d0=11 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,11,3) */
s_mov_b32 s54, 91                                  // coordOffset0 d0=11 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,12,0) */
s_mov_b32 s54, 96                                  // coordOffset0 d0=12 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 10, 0, 2), (0, 10, 0, 3), (0, 11, 0, 0), (0, 11, 0, 1), (0, 11, 0, 2), (0, 11, 0, 3), (0, 12, 0, 0)] */
v_mul_f32 v[vgprValuC+42], s[sgprAlpha], v[vgprValuC+42] // *= alpha
v_mul_f32 v[vgprValuC+43], s[sgprAlpha], v[vgprValuC+43] // *= alpha
v_mul_f32 v[vgprValuC+44], s[sgprAlpha], v[vgprValuC+44] // *= alpha
v_mul_f32 v[vgprValuC+45], s[sgprAlpha], v[vgprValuC+45] // *= alpha
v_mul_f32 v[vgprValuC+46], s[sgprAlpha], v[vgprValuC+46] // *= alpha
v_mul_f32 v[vgprValuC+47], s[sgprAlpha], v[vgprValuC+47] // *= alpha
v_mul_f32 v[vgprValuC+48], s[sgprAlpha], v[vgprValuC+48] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+42]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+43]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+44]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+45]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+46]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+47]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+48]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0037                         // if exec is zero skip loop

/* atomic CAS loop */
label_0036:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+42]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+43]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+44]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+45]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+46]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+47]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+48]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0036                        // try again if not complete
label_0037:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #7 (d1,d0,vc1,vc0) = */
/*    (0,12,0,1:vw1:vaw:1); (0,12,0,2:vw1:vaw:1); (0,12,0,3:vw1:vaw:1); (0,13,0,0:vw1:vaw:1); (0,13,0,1:vw1:vaw:1); (0,13,0,2:vw1:vaw:1); (0,13,0,3:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,12,1) */
s_mov_b32 s54, 97                                  // coordOffset0 d0=12 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,12,2) */
s_mov_b32 s54, 98                                  // coordOffset0 d0=12 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,12,3) */
s_mov_b32 s54, 99                                  // coordOffset0 d0=12 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,13,0) */
s_mov_b32 s54, 104                                 // coordOffset0 d0=13 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,13,1) */
s_mov_b32 s54, 105                                 // coordOffset0 d0=13 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,13,2) */
s_mov_b32 s54, 106                                 // coordOffset0 d0=13 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,13,3) */
s_mov_b32 s54, 107                                 // coordOffset0 d0=13 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 12, 0, 1), (0, 12, 0, 2), (0, 12, 0, 3), (0, 13, 0, 0), (0, 13, 0, 1), (0, 13, 0, 2), (0, 13, 0, 3)] */
v_mul_f32 v[vgprValuC+49], s[sgprAlpha], v[vgprValuC+49] // *= alpha
v_mul_f32 v[vgprValuC+50], s[sgprAlpha], v[vgprValuC+50] // *= alpha
v_mul_f32 v[vgprValuC+51], s[sgprAlpha], v[vgprValuC+51] // *= alpha
v_mul_f32 v[vgprValuC+52], s[sgprAlpha], v[vgprValuC+52] // *= alpha
v_mul_f32 v[vgprValuC+53], s[sgprAlpha], v[vgprValuC+53] // *= alpha
v_mul_f32 v[vgprValuC+54], s[sgprAlpha], v[vgprValuC+54] // *= alpha
v_mul_f32 v[vgprValuC+55], s[sgprAlpha], v[vgprValuC+55] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+49]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+50]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+51]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+52]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+53]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+54]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+55]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0039                         // if exec is zero skip loop

/* atomic CAS loop */
label_0038:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+49]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+50]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+51]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+52]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+53]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+54]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+55]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0038                        // try again if not complete
label_0039:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #8 (d1,d0,vc1,vc0) = */
/*    (0,14,0,0:vw1:vaw:1); (0,14,0,1:vw1:vaw:1); (0,14,0,2:vw1:vaw:1); (0,14,0,3:vw1:vaw:1); (0,15,0,0:vw1:vaw:1); (0,15,0,1:vw1:vaw:1); (0,15,0,2:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,14,0) */
s_mov_b32 s54, 112                                 // coordOffset0 d0=14 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,14,1) */
s_mov_b32 s54, 113                                 // coordOffset0 d0=14 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,14,2) */
s_mov_b32 s54, 114                                 // coordOffset0 d0=14 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,14,3) */
s_mov_b32 s54, 115                                 // coordOffset0 d0=14 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,15,0) */
s_mov_b32 s54, 120                                 // coordOffset0 d0=15 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,15,1) */
s_mov_b32 s54, 121                                 // coordOffset0 d0=15 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,15,2) */
s_mov_b32 s54, 122                                 // coordOffset0 d0=15 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 14, 0, 0), (0, 14, 0, 1), (0, 14, 0, 2), (0, 14, 0, 3), (0, 15, 0, 0), (0, 15, 0, 1), (0, 15, 0, 2)] */
v_mul_f32 v[vgprValuC+56], s[sgprAlpha], v[vgprValuC+56] // *= alpha
v_mul_f32 v[vgprValuC+57], s[sgprAlpha], v[vgprValuC+57] // *= alpha
v_mul_f32 v[vgprValuC+58], s[sgprAlpha], v[vgprValuC+58] // *= alpha
v_mul_f32 v[vgprValuC+59], s[sgprAlpha], v[vgprValuC+59] // *= alpha
v_mul_f32 v[vgprValuC+60], s[sgprAlpha], v[vgprValuC+60] // *= alpha
v_mul_f32 v[vgprValuC+61], s[sgprAlpha], v[vgprValuC+61] // *= alpha
v_mul_f32 v[vgprValuC+62], s[sgprAlpha], v[vgprValuC+62] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+56]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+57]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+58]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+59]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+60]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+61]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+62]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0041                         // if exec is zero skip loop

/* atomic CAS loop */
label_0040:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+56]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+57]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+58]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+59]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+60]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+61]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+62]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0040                        // try again if not complete
label_0041:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #9 (d1,d0,vc1,vc0) = */
/*    (0,15,0,3:vw1:vaw:1); (0,16,0,0:vw1:vaw:1); (0,16,0,1:vw1:vaw:1); (0,16,0,2:vw1:vaw:1); (0,16,0,3:vw1:vaw:1); (0,17,0,0:vw1:vaw:1); (0,17,0,1:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,15,3) */
s_mov_b32 s54, 123                                 // coordOffset0 d0=15 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,16,0) */
s_mov_b32 s54, 128                                 // coordOffset0 d0=16 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,16,1) */
s_mov_b32 s54, 129                                 // coordOffset0 d0=16 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,16,2) */
s_mov_b32 s54, 130                                 // coordOffset0 d0=16 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,16,3) */
s_mov_b32 s54, 131                                 // coordOffset0 d0=16 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,17,0) */
s_mov_b32 s54, 136                                 // coordOffset0 d0=17 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,17,1) */
s_mov_b32 s54, 137                                 // coordOffset0 d0=17 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 15, 0, 3), (0, 16, 0, 0), (0, 16, 0, 1), (0, 16, 0, 2), (0, 16, 0, 3), (0, 17, 0, 0), (0, 17, 0, 1)] */
v_mul_f32 v[vgprValuC+63], s[sgprAlpha], v[vgprValuC+63] // *= alpha
v_mul_f32 v[vgprValuC+64], s[sgprAlpha], v[vgprValuC+64] // *= alpha
v_mul_f32 v[vgprValuC+65], s[sgprAlpha], v[vgprValuC+65] // *= alpha
v_mul_f32 v[vgprValuC+66], s[sgprAlpha], v[vgprValuC+66] // *= alpha
v_mul_f32 v[vgprValuC+67], s[sgprAlpha], v[vgprValuC+67] // *= alpha
v_mul_f32 v[vgprValuC+68], s[sgprAlpha], v[vgprValuC+68] // *= alpha
v_mul_f32 v[vgprValuC+69], s[sgprAlpha], v[vgprValuC+69] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+63]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+64]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+65]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+66]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+67]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+68]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+69]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0043                         // if exec is zero skip loop

/* atomic CAS loop */
label_0042:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+63]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+64]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+65]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+66]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+67]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+68]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+69]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0042                        // try again if not complete
label_0043:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #10 (d1,d0,vc1,vc0) = */
/*    (0,17,0,2:vw1:vaw:1); (0,17,0,3:vw1:vaw:1); (0,18,0,0:vw1:vaw:1); (0,18,0,1:vw1:vaw:1); (0,18,0,2:vw1:vaw:1); (0,18,0,3:vw1:vaw:1); (0,19,0,0:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,17,2) */
s_mov_b32 s54, 138                                 // coordOffset0 d0=17 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,17,3) */
s_mov_b32 s54, 139                                 // coordOffset0 d0=17 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,18,0) */
s_mov_b32 s54, 144                                 // coordOffset0 d0=18 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,18,1) */
s_mov_b32 s54, 145                                 // coordOffset0 d0=18 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,18,2) */
s_mov_b32 s54, 146                                 // coordOffset0 d0=18 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,18,3) */
s_mov_b32 s54, 147                                 // coordOffset0 d0=18 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,19,0) */
s_mov_b32 s54, 152                                 // coordOffset0 d0=19 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 17, 0, 2), (0, 17, 0, 3), (0, 18, 0, 0), (0, 18, 0, 1), (0, 18, 0, 2), (0, 18, 0, 3), (0, 19, 0, 0)] */
v_mul_f32 v[vgprValuC+70], s[sgprAlpha], v[vgprValuC+70] // *= alpha
v_mul_f32 v[vgprValuC+71], s[sgprAlpha], v[vgprValuC+71] // *= alpha
v_mul_f32 v[vgprValuC+72], s[sgprAlpha], v[vgprValuC+72] // *= alpha
v_mul_f32 v[vgprValuC+73], s[sgprAlpha], v[vgprValuC+73] // *= alpha
v_mul_f32 v[vgprValuC+74], s[sgprAlpha], v[vgprValuC+74] // *= alpha
v_mul_f32 v[vgprValuC+75], s[sgprAlpha], v[vgprValuC+75] // *= alpha
v_mul_f32 v[vgprValuC+76], s[sgprAlpha], v[vgprValuC+76] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+70]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+71]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+72]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+73]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+74]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+75]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+76]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0045                         // if exec is zero skip loop

/* atomic CAS loop */
label_0044:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+70]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+71]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+72]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+73]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+74]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+75]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+76]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0044                        // try again if not complete
label_0045:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #11 (d1,d0,vc1,vc0) = */
/*    (0,19,0,1:vw1:vaw:1); (0,19,0,2:vw1:vaw:1); (0,19,0,3:vw1:vaw:1); (0,20,0,0:vw1:vaw:1); (0,20,0,1:vw1:vaw:1); (0,20,0,2:vw1:vaw:1); (0,20,0,3:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,19,1) */
s_mov_b32 s54, 153                                 // coordOffset0 d0=19 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,19,2) */
s_mov_b32 s54, 154                                 // coordOffset0 d0=19 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,19,3) */
s_mov_b32 s54, 155                                 // coordOffset0 d0=19 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,20,0) */
s_mov_b32 s54, 160                                 // coordOffset0 d0=20 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,20,1) */
s_mov_b32 s54, 161                                 // coordOffset0 d0=20 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,20,2) */
s_mov_b32 s54, 162                                 // coordOffset0 d0=20 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,20,3) */
s_mov_b32 s54, 163                                 // coordOffset0 d0=20 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 19, 0, 1), (0, 19, 0, 2), (0, 19, 0, 3), (0, 20, 0, 0), (0, 20, 0, 1), (0, 20, 0, 2), (0, 20, 0, 3)] */
v_mul_f32 v[vgprValuC+77], s[sgprAlpha], v[vgprValuC+77] // *= alpha
v_mul_f32 v[vgprValuC+78], s[sgprAlpha], v[vgprValuC+78] // *= alpha
v_mul_f32 v[vgprValuC+79], s[sgprAlpha], v[vgprValuC+79] // *= alpha
v_mul_f32 v[vgprValuC+80], s[sgprAlpha], v[vgprValuC+80] // *= alpha
v_mul_f32 v[vgprValuC+81], s[sgprAlpha], v[vgprValuC+81] // *= alpha
v_mul_f32 v[vgprValuC+82], s[sgprAlpha], v[vgprValuC+82] // *= alpha
v_mul_f32 v[vgprValuC+83], s[sgprAlpha], v[vgprValuC+83] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+77]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+78]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+79]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+80]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+81]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+82]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+83]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0047                         // if exec is zero skip loop

/* atomic CAS loop */
label_0046:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+77]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+78]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+79]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+80]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+81]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+82]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+83]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0046                        // try again if not complete
label_0047:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #12 (d1,d0,vc1,vc0) = */
/*    (0,21,0,0:vw1:vaw:1); (0,21,0,1:vw1:vaw:1); (0,21,0,2:vw1:vaw:1); (0,21,0,3:vw1:vaw:1); (0,22,0,0:vw1:vaw:1); (0,22,0,1:vw1:vaw:1); (0,22,0,2:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,21,0) */
s_mov_b32 s54, 168                                 // coordOffset0 d0=21 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,21,1) */
s_mov_b32 s54, 169                                 // coordOffset0 d0=21 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,21,2) */
s_mov_b32 s54, 170                                 // coordOffset0 d0=21 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,21,3) */
s_mov_b32 s54, 171                                 // coordOffset0 d0=21 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,22,0) */
s_mov_b32 s54, 176                                 // coordOffset0 d0=22 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,22,1) */
s_mov_b32 s54, 177                                 // coordOffset0 d0=22 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,22,2) */
s_mov_b32 s54, 178                                 // coordOffset0 d0=22 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 21, 0, 0), (0, 21, 0, 1), (0, 21, 0, 2), (0, 21, 0, 3), (0, 22, 0, 0), (0, 22, 0, 1), (0, 22, 0, 2)] */
v_mul_f32 v[vgprValuC+84], s[sgprAlpha], v[vgprValuC+84] // *= alpha
v_mul_f32 v[vgprValuC+85], s[sgprAlpha], v[vgprValuC+85] // *= alpha
v_mul_f32 v[vgprValuC+86], s[sgprAlpha], v[vgprValuC+86] // *= alpha
v_mul_f32 v[vgprValuC+87], s[sgprAlpha], v[vgprValuC+87] // *= alpha
v_mul_f32 v[vgprValuC+88], s[sgprAlpha], v[vgprValuC+88] // *= alpha
v_mul_f32 v[vgprValuC+89], s[sgprAlpha], v[vgprValuC+89] // *= alpha
v_mul_f32 v[vgprValuC+90], s[sgprAlpha], v[vgprValuC+90] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+84]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+85]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+86]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+87]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+88]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+89]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+90]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0049                         // if exec is zero skip loop

/* atomic CAS loop */
label_0048:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+84]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+85]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+86]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+87]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+88]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+89]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+90]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0048                        // try again if not complete
label_0049:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #13 (d1,d0,vc1,vc0) = */
/*    (0,22,0,3:vw1:vaw:1); (0,23,0,0:vw1:vaw:1); (0,23,0,1:vw1:vaw:1); (0,23,0,2:vw1:vaw:1); (0,23,0,3:vw1:vaw:1); (0,24,0,0:vw1:vaw:1); (0,24,0,1:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,22,3) */
s_mov_b32 s54, 179                                 // coordOffset0 d0=22 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,23,0) */
s_mov_b32 s54, 184                                 // coordOffset0 d0=23 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,23,1) */
s_mov_b32 s54, 185                                 // coordOffset0 d0=23 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,23,2) */
s_mov_b32 s54, 186                                 // coordOffset0 d0=23 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,23,3) */
s_mov_b32 s54, 187                                 // coordOffset0 d0=23 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,24,0) */
s_mov_b32 s54, 192                                 // coordOffset0 d0=24 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,24,1) */
s_mov_b32 s54, 193                                 // coordOffset0 d0=24 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 22, 0, 3), (0, 23, 0, 0), (0, 23, 0, 1), (0, 23, 0, 2), (0, 23, 0, 3), (0, 24, 0, 0), (0, 24, 0, 1)] */
v_mul_f32 v[vgprValuC+91], s[sgprAlpha], v[vgprValuC+91] // *= alpha
v_mul_f32 v[vgprValuC+92], s[sgprAlpha], v[vgprValuC+92] // *= alpha
v_mul_f32 v[vgprValuC+93], s[sgprAlpha], v[vgprValuC+93] // *= alpha
v_mul_f32 v[vgprValuC+94], s[sgprAlpha], v[vgprValuC+94] // *= alpha
v_mul_f32 v[vgprValuC+95], s[sgprAlpha], v[vgprValuC+95] // *= alpha
v_mul_f32 v[vgprValuC+96], s[sgprAlpha], v[vgprValuC+96] // *= alpha
v_mul_f32 v[vgprValuC+97], s[sgprAlpha], v[vgprValuC+97] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+91]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+92]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+93]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+94]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+95]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+96]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+97]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0051                         // if exec is zero skip loop

/* atomic CAS loop */
label_0050:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+91]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+92]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+93]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+94]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+95]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+96]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+97]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0050                        // try again if not complete
label_0051:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #14 (d1,d0,vc1,vc0) = */
/*    (0,24,0,2:vw1:vaw:1); (0,24,0,3:vw1:vaw:1); (0,25,0,0:vw1:vaw:1); (0,25,0,1:vw1:vaw:1); (0,25,0,2:vw1:vaw:1); (0,25,0,3:vw1:vaw:1); (0,26,0,0:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,24,2) */
s_mov_b32 s54, 194                                 // coordOffset0 d0=24 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,24,3) */
s_mov_b32 s54, 195                                 // coordOffset0 d0=24 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,25,0) */
s_mov_b32 s54, 200                                 // coordOffset0 d0=25 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,25,1) */
s_mov_b32 s54, 201                                 // coordOffset0 d0=25 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,25,2) */
s_mov_b32 s54, 202                                 // coordOffset0 d0=25 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,25,3) */
s_mov_b32 s54, 203                                 // coordOffset0 d0=25 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,26,0) */
s_mov_b32 s54, 208                                 // coordOffset0 d0=26 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 24, 0, 2), (0, 24, 0, 3), (0, 25, 0, 0), (0, 25, 0, 1), (0, 25, 0, 2), (0, 25, 0, 3), (0, 26, 0, 0)] */
v_mul_f32 v[vgprValuC+98], s[sgprAlpha], v[vgprValuC+98] // *= alpha
v_mul_f32 v[vgprValuC+99], s[sgprAlpha], v[vgprValuC+99] // *= alpha
v_mul_f32 v[vgprValuC+100], s[sgprAlpha], v[vgprValuC+100] // *= alpha
v_mul_f32 v[vgprValuC+101], s[sgprAlpha], v[vgprValuC+101] // *= alpha
v_mul_f32 v[vgprValuC+102], s[sgprAlpha], v[vgprValuC+102] // *= alpha
v_mul_f32 v[vgprValuC+103], s[sgprAlpha], v[vgprValuC+103] // *= alpha
v_mul_f32 v[vgprValuC+104], s[sgprAlpha], v[vgprValuC+104] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+98]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+99]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+100]               // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+101]               // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+102]               // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+103]               // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+104]               // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0053                         // if exec is zero skip loop

/* atomic CAS loop */
label_0052:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+98]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+99]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+100]               // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+101]               // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+102]               // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+103]               // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+104]               // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0052                        // try again if not complete
label_0053:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #15 (d1,d0,vc1,vc0) = */
/*    (0,26,0,1:vw1:vaw:1); (0,26,0,2:vw1:vaw:1); (0,26,0,3:vw1:vaw:1); (0,27,0,0:vw1:vaw:1); (0,27,0,1:vw1:vaw:1); (0,27,0,2:vw1:vaw:1); (0,27,0,3:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,26,1) */
s_mov_b32 s54, 209                                 // coordOffset0 d0=26 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,26,2) */
s_mov_b32 s54, 210                                 // coordOffset0 d0=26 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,26,3) */
s_mov_b32 s54, 211                                 // coordOffset0 d0=26 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,27,0) */
s_mov_b32 s54, 216                                 // coordOffset0 d0=27 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,27,1) */
s_mov_b32 s54, 217                                 // coordOffset0 d0=27 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,27,2) */
s_mov_b32 s54, 218                                 // coordOffset0 d0=27 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,27,3) */
s_mov_b32 s54, 219                                 // coordOffset0 d0=27 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 26, 0, 1), (0, 26, 0, 2), (0, 26, 0, 3), (0, 27, 0, 0), (0, 27, 0, 1), (0, 27, 0, 2), (0, 27, 0, 3)] */
v_mul_f32 v[vgprValuC+105], s[sgprAlpha], v[vgprValuC+105] // *= alpha
v_mul_f32 v[vgprValuC+106], s[sgprAlpha], v[vgprValuC+106] // *= alpha
v_mul_f32 v[vgprValuC+107], s[sgprAlpha], v[vgprValuC+107] // *= alpha
v_mul_f32 v[vgprValuC+108], s[sgprAlpha], v[vgprValuC+108] // *= alpha
v_mul_f32 v[vgprValuC+109], s[sgprAlpha], v[vgprValuC+109] // *= alpha
v_mul_f32 v[vgprValuC+110], s[sgprAlpha], v[vgprValuC+110] // *= alpha
v_mul_f32 v[vgprValuC+111], s[sgprAlpha], v[vgprValuC+111] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+105]               // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+106]               // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+107]               // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+108]               // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+109]               // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+110]               // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+111]               // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0055                         // if exec is zero skip loop

/* atomic CAS loop */
label_0054:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+105]               // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+106]               // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+107]               // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+108]               // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+109]               // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+110]               // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+111]               // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0054                        // try again if not complete
label_0055:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #16 (d1,d0,vc1,vc0) = */
/*    (0,28,0,0:vw1:vaw:1); (0,28,0,1:vw1:vaw:1); (0,28,0,2:vw1:vaw:1); (0,28,0,3:vw1:vaw:1); (0,29,0,0:vw1:vaw:1); (0,29,0,1:vw1:vaw:1); (0,29,0,2:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,28,0) */
s_mov_b32 s54, 224                                 // coordOffset0 d0=28 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,28,1) */
s_mov_b32 s54, 225                                 // coordOffset0 d0=28 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,28,2) */
s_mov_b32 s54, 226                                 // coordOffset0 d0=28 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,28,3) */
s_mov_b32 s54, 227                                 // coordOffset0 d0=28 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,29,0) */
s_mov_b32 s54, 232                                 // coordOffset0 d0=29 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,29,1) */
s_mov_b32 s54, 233                                 // coordOffset0 d0=29 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,29,2) */
s_mov_b32 s54, 234                                 // coordOffset0 d0=29 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 28, 0, 0), (0, 28, 0, 1), (0, 28, 0, 2), (0, 28, 0, 3), (0, 29, 0, 0), (0, 29, 0, 1), (0, 29, 0, 2)] */
v_mul_f32 v[vgprValuC+112], s[sgprAlpha], v[vgprValuC+112] // *= alpha
v_mul_f32 v[vgprValuC+113], s[sgprAlpha], v[vgprValuC+113] // *= alpha
v_mul_f32 v[vgprValuC+114], s[sgprAlpha], v[vgprValuC+114] // *= alpha
v_mul_f32 v[vgprValuC+115], s[sgprAlpha], v[vgprValuC+115] // *= alpha
v_mul_f32 v[vgprValuC+116], s[sgprAlpha], v[vgprValuC+116] // *= alpha
v_mul_f32 v[vgprValuC+117], s[sgprAlpha], v[vgprValuC+117] // *= alpha
v_mul_f32 v[vgprValuC+118], s[sgprAlpha], v[vgprValuC+118] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+112]               // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+113]               // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+114]               // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+115]               // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+116]               // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+117]               // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+118]               // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0057                         // if exec is zero skip loop

/* atomic CAS loop */
label_0056:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+112]               // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+113]               // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+114]               // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+115]               // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+116]               // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+117]               // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+118]               // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0056                        // try again if not complete
label_0057:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #17 (d1,d0,vc1,vc0) = */
/*    (0,29,0,3:vw1:vaw:1); (0,30,0,0:vw1:vaw:1); (0,30,0,1:vw1:vaw:1); (0,30,0,2:vw1:vaw:1); (0,30,0,3:vw1:vaw:1); (0,31,0,0:vw1:vaw:1); (0,31,0,1:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,29,3) */
s_mov_b32 s54, 235                                 // coordOffset0 d0=29 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,30,0) */
s_mov_b32 s54, 240                                 // coordOffset0 d0=30 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,30,1) */
s_mov_b32 s54, 241                                 // coordOffset0 d0=30 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,30,2) */
s_mov_b32 s54, 242                                 // coordOffset0 d0=30 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,30,3) */
s_mov_b32 s54, 243                                 // coordOffset0 d0=30 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,31,0) */
s_mov_b32 s54, 248                                 // coordOffset0 d0=31 vc0=0
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,31,1) */
s_mov_b32 s54, 249                                 // coordOffset0 d0=31 vc0=1
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 29, 0, 3), (0, 30, 0, 0), (0, 30, 0, 1), (0, 30, 0, 2), (0, 30, 0, 3), (0, 31, 0, 0), (0, 31, 0, 1)] */
v_mul_f32 v[vgprValuC+119], s[sgprAlpha], v[vgprValuC+119] // *= alpha
v_mul_f32 v[vgprValuC+120], s[sgprAlpha], v[vgprValuC+120] // *= alpha
v_mul_f32 v[vgprValuC+121], s[sgprAlpha], v[vgprValuC+121] // *= alpha
v_mul_f32 v[vgprValuC+122], s[sgprAlpha], v[vgprValuC+122] // *= alpha
v_mul_f32 v[vgprValuC+123], s[sgprAlpha], v[vgprValuC+123] // *= alpha
v_mul_f32 v[vgprValuC+124], s[sgprAlpha], v[vgprValuC+124] // *= alpha
v_mul_f32 v[vgprValuC+125], s[sgprAlpha], v[vgprValuC+125] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+119]               // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+120]               // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v46, v47, v[vgprValuC+121]               // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v49, v50, v[vgprValuC+122]               // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v52, v53, v[vgprValuC+123]               // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v55, v56, v[vgprValuC+124]               // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v58, v59, v[vgprValuC+125]               // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[64:65], v46, v47                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[66:67], v49, v50                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[68:69], v52, v53                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[70:71], v55, v56                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[72:73], v58, v59                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0059                         // if exec is zero skip loop

/* atomic CAS loop */
label_0058:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+119]               // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+120]               // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+121]               // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+122]               // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+123]               // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+124]               // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+125]               // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0058                        // try again if not complete
label_0059:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Batch #18 (d1,d0,vc1,vc0) = */
/*    (0,31,0,2:vw1:vaw:1); (0,31,0,3:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,31,2) */
s_mov_b32 s54, 250                                 // coordOffset0 d0=31 vc0=2
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,31,3) */
s_mov_b32 s54, 251                                 // coordOffset0 d0=31 vc0=3
_v_add_co_u32 v35, vcc, v32, s54                   // coord0.2: coord0 += d0*sg0*VW + vc0
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 31, 0, 2), (0, 31, 0, 3)] */
v_mul_f32 v[vgprValuC+126], s[sgprAlpha], v[vgprValuC+126] // *= alpha
v_mul_f32 v[vgprValuC+127], s[sgprAlpha], v[vgprValuC+127] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
v_add_f32 v38, v39, v[vgprValuC+126]               // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
v_add_f32 v43, v44, v[vgprValuC+127]               // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
v_cmp_ne_u32 s[60:61], v38, v39                    // c read during atomic != c read during prior load
v_cmp_ne_u32 s[62:63], v43, v44                    // c read during atomic != c read during prior load

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0061                         // if exec is zero skip loop

/* atomic CAS loop */
label_0060:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+126]               // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+127]               // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0060                        // try again if not complete
label_0061:
s_mov_b64 exec, -1                                 // full mask -> exec
s_branch label_0023                                // jump to end
GW_B0_E1_21:

/* allocate 20 sgpr. perBatch=6 perElement=2 elements=7 */
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #0 (d1,d0,vc1,vc0) = */
/*    (0,0,0,0:vw1:vaw:1); (0,0,0,1:vw1:vaw:1); (0,0,0,2:vw1:vaw:1); (0,0,0,3:vw1:vaw:1); (0,1,0,0:vw1:vaw:1); (0,1,0,1:vw1:vaw:1); (0,1,0,2:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,0,0) */
v_cmp_lt_u32 s[54:55], v32, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v37, v34, v32, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, -1, v37, s[60:61]               // clip if OOB. offset
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,0,1) */
_v_add_co_u32 v35, vcc, v32, 1                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, -1, v42, s[62:63]               // clip if OOB. offset
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,0,2) */
_v_add_co_u32 v35, vcc, v32, 2                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, -1, v45, s[64:65]               // clip if OOB. offset
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,0,3) */
_v_add_co_u32 v35, vcc, v32, 3                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, -1, v48, s[66:67]               // clip if OOB. offset
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,1,0) */
_v_add_co_u32 v35, vcc, v32, 8                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, -1, v51, s[68:69]               // clip if OOB. offset
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,1,1) */
_v_add_co_u32 v35, vcc, v32, 9                     // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v54, -1, v54, s[70:71]               // clip if OOB. offset
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,1,2) */
_v_add_co_u32 v35, vcc, v32, 10                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v57, -1, v57, s[72:73]               // clip if OOB. offset
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 0, 2)] */
v_mul_f32 v[vgprValuC+0], s[sgprAlpha], v[vgprValuC+0] // *= alpha
v_mul_f32 v[vgprValuC+1], s[sgprAlpha], v[vgprValuC+1] // *= alpha
v_mul_f32 v[vgprValuC+2], s[sgprAlpha], v[vgprValuC+2] // *= alpha
v_mul_f32 v[vgprValuC+3], s[sgprAlpha], v[vgprValuC+3] // *= alpha
v_mul_f32 v[vgprValuC+4], s[sgprAlpha], v[vgprValuC+4] // *= alpha
v_mul_f32 v[vgprValuC+5], s[sgprAlpha], v[vgprValuC+5] // *= alpha
v_mul_f32 v[vgprValuC+6], s[sgprAlpha], v[vgprValuC+6] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec (before atomic)
v_add_f32 v38, v39, v[vgprValuC+0]                 // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[62:63]                           // sgprs -> exec (before atomic)
v_add_f32 v43, v44, v[vgprValuC+1]                 // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[64:65]                           // sgprs -> exec (before atomic)
v_add_f32 v46, v47, v[vgprValuC+2]                 // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[66:67]                           // sgprs -> exec (before atomic)
v_add_f32 v49, v50, v[vgprValuC+3]                 // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[68:69]                           // sgprs -> exec (before atomic)
v_add_f32 v52, v53, v[vgprValuC+4]                 // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[70:71]                           // sgprs -> exec (before atomic)
v_add_f32 v55, v56, v[vgprValuC+5]                 // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[72:73]                           // sgprs -> exec (before atomic)
v_add_f32 v58, v59, v[vgprValuC+6]                 // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v38, v39                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v43, v44                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v46, v47                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v49, v50                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v52, v53                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v55, v56                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v58, v59                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0063                         // if exec is zero skip loop

/* atomic CAS loop */
label_0062:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+0]                 // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+1]                 // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+2]                 // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+3]                 // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+4]                 // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+5]                 // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+6]                 // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0062                        // try again if not complete
label_0063:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #1 (d1,d0,vc1,vc0) = */
/*    (0,1,0,3:vw1:vaw:1); (0,2,0,0:vw1:vaw:1); (0,2,0,1:vw1:vaw:1); (0,2,0,2:vw1:vaw:1); (0,2,0,3:vw1:vaw:1); (0,3,0,0:vw1:vaw:1); (0,3,0,1:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,1,3) */
_v_add_co_u32 v35, vcc, v32, 11                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, -1, v37, s[60:61]               // clip if OOB. offset
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,2,0) */
_v_add_co_u32 v35, vcc, v32, 16                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, -1, v42, s[62:63]               // clip if OOB. offset
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,2,1) */
_v_add_co_u32 v35, vcc, v32, 17                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, -1, v45, s[64:65]               // clip if OOB. offset
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,2,2) */
_v_add_co_u32 v35, vcc, v32, 18                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, -1, v48, s[66:67]               // clip if OOB. offset
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,2,3) */
_v_add_co_u32 v35, vcc, v32, 19                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, -1, v51, s[68:69]               // clip if OOB. offset
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,3,0) */
_v_add_co_u32 v35, vcc, v32, 24                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v54, -1, v54, s[70:71]               // clip if OOB. offset
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,3,1) */
_v_add_co_u32 v35, vcc, v32, 25                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v57, -1, v57, s[72:73]               // clip if OOB. offset
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 1, 0, 3), (0, 2, 0, 0), (0, 2, 0, 1), (0, 2, 0, 2), (0, 2, 0, 3), (0, 3, 0, 0), (0, 3, 0, 1)] */
v_mul_f32 v[vgprValuC+7], s[sgprAlpha], v[vgprValuC+7] // *= alpha
v_mul_f32 v[vgprValuC+8], s[sgprAlpha], v[vgprValuC+8] // *= alpha
v_mul_f32 v[vgprValuC+9], s[sgprAlpha], v[vgprValuC+9] // *= alpha
v_mul_f32 v[vgprValuC+10], s[sgprAlpha], v[vgprValuC+10] // *= alpha
v_mul_f32 v[vgprValuC+11], s[sgprAlpha], v[vgprValuC+11] // *= alpha
v_mul_f32 v[vgprValuC+12], s[sgprAlpha], v[vgprValuC+12] // *= alpha
v_mul_f32 v[vgprValuC+13], s[sgprAlpha], v[vgprValuC+13] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec (before atomic)
v_add_f32 v38, v39, v[vgprValuC+7]                 // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[62:63]                           // sgprs -> exec (before atomic)
v_add_f32 v43, v44, v[vgprValuC+8]                 // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[64:65]                           // sgprs -> exec (before atomic)
v_add_f32 v46, v47, v[vgprValuC+9]                 // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[66:67]                           // sgprs -> exec (before atomic)
v_add_f32 v49, v50, v[vgprValuC+10]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[68:69]                           // sgprs -> exec (before atomic)
v_add_f32 v52, v53, v[vgprValuC+11]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[70:71]                           // sgprs -> exec (before atomic)
v_add_f32 v55, v56, v[vgprValuC+12]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[72:73]                           // sgprs -> exec (before atomic)
v_add_f32 v58, v59, v[vgprValuC+13]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v38, v39                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v43, v44                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v46, v47                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v49, v50                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v52, v53                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v55, v56                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v58, v59                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0065                         // if exec is zero skip loop

/* atomic CAS loop */
label_0064:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+7]                 // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+8]                 // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+9]                 // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+10]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+11]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+12]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+13]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0064                        // try again if not complete
label_0065:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #2 (d1,d0,vc1,vc0) = */
/*    (0,3,0,2:vw1:vaw:1); (0,3,0,3:vw1:vaw:1); (0,4,0,0:vw1:vaw:1); (0,4,0,1:vw1:vaw:1); (0,4,0,2:vw1:vaw:1); (0,4,0,3:vw1:vaw:1); (0,5,0,0:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,3,2) */
_v_add_co_u32 v35, vcc, v32, 26                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, -1, v37, s[60:61]               // clip if OOB. offset
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,3,3) */
_v_add_co_u32 v35, vcc, v32, 27                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, -1, v42, s[62:63]               // clip if OOB. offset
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,4,0) */
_v_add_co_u32 v35, vcc, v32, 32                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, -1, v45, s[64:65]               // clip if OOB. offset
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,4,1) */
_v_add_co_u32 v35, vcc, v32, 33                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, -1, v48, s[66:67]               // clip if OOB. offset
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,4,2) */
_v_add_co_u32 v35, vcc, v32, 34                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, -1, v51, s[68:69]               // clip if OOB. offset
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,4,3) */
_v_add_co_u32 v35, vcc, v32, 35                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v54, -1, v54, s[70:71]               // clip if OOB. offset
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,5,0) */
_v_add_co_u32 v35, vcc, v32, 40                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v57, -1, v57, s[72:73]               // clip if OOB. offset
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 3, 0, 2), (0, 3, 0, 3), (0, 4, 0, 0), (0, 4, 0, 1), (0, 4, 0, 2), (0, 4, 0, 3), (0, 5, 0, 0)] */
v_mul_f32 v[vgprValuC+14], s[sgprAlpha], v[vgprValuC+14] // *= alpha
v_mul_f32 v[vgprValuC+15], s[sgprAlpha], v[vgprValuC+15] // *= alpha
v_mul_f32 v[vgprValuC+16], s[sgprAlpha], v[vgprValuC+16] // *= alpha
v_mul_f32 v[vgprValuC+17], s[sgprAlpha], v[vgprValuC+17] // *= alpha
v_mul_f32 v[vgprValuC+18], s[sgprAlpha], v[vgprValuC+18] // *= alpha
v_mul_f32 v[vgprValuC+19], s[sgprAlpha], v[vgprValuC+19] // *= alpha
v_mul_f32 v[vgprValuC+20], s[sgprAlpha], v[vgprValuC+20] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec (before atomic)
v_add_f32 v38, v39, v[vgprValuC+14]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[62:63]                           // sgprs -> exec (before atomic)
v_add_f32 v43, v44, v[vgprValuC+15]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[64:65]                           // sgprs -> exec (before atomic)
v_add_f32 v46, v47, v[vgprValuC+16]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[66:67]                           // sgprs -> exec (before atomic)
v_add_f32 v49, v50, v[vgprValuC+17]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[68:69]                           // sgprs -> exec (before atomic)
v_add_f32 v52, v53, v[vgprValuC+18]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[70:71]                           // sgprs -> exec (before atomic)
v_add_f32 v55, v56, v[vgprValuC+19]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[72:73]                           // sgprs -> exec (before atomic)
v_add_f32 v58, v59, v[vgprValuC+20]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v38, v39                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v43, v44                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v46, v47                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v49, v50                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v52, v53                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v55, v56                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v58, v59                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0067                         // if exec is zero skip loop

/* atomic CAS loop */
label_0066:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+14]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+15]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+16]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+17]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+18]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+19]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+20]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0066                        // try again if not complete
label_0067:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #3 (d1,d0,vc1,vc0) = */
/*    (0,5,0,1:vw1:vaw:1); (0,5,0,2:vw1:vaw:1); (0,5,0,3:vw1:vaw:1); (0,6,0,0:vw1:vaw:1); (0,6,0,1:vw1:vaw:1); (0,6,0,2:vw1:vaw:1); (0,6,0,3:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,5,1) */
_v_add_co_u32 v35, vcc, v32, 41                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, -1, v37, s[60:61]               // clip if OOB. offset
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,5,2) */
_v_add_co_u32 v35, vcc, v32, 42                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, -1, v42, s[62:63]               // clip if OOB. offset
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,5,3) */
_v_add_co_u32 v35, vcc, v32, 43                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, -1, v45, s[64:65]               // clip if OOB. offset
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,6,0) */
_v_add_co_u32 v35, vcc, v32, 48                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, -1, v48, s[66:67]               // clip if OOB. offset
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,6,1) */
_v_add_co_u32 v35, vcc, v32, 49                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[68:69], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[68:69], s[54:55], s[68:69]             // in0 && in1
_v_add_lshl_u32 v51, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v51, -1, v51, s[68:69]               // clip if OOB. offset
buffer_load_dword v53, v51, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,6,2) */
_v_add_co_u32 v35, vcc, v32, 50                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[70:71], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[70:71], s[54:55], s[70:71]             // in0 && in1
_v_add_lshl_u32 v54, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v54, -1, v54, s[70:71]               // clip if OOB. offset
buffer_load_dword v56, v54, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,6,3) */
_v_add_co_u32 v35, vcc, v32, 51                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[72:73], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[72:73], s[54:55], s[72:73]             // in0 && in1
_v_add_lshl_u32 v57, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v57, -1, v57, s[72:73]               // clip if OOB. offset
buffer_load_dword v59, v57, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 5, 0, 1), (0, 5, 0, 2), (0, 5, 0, 3), (0, 6, 0, 0), (0, 6, 0, 1), (0, 6, 0, 2), (0, 6, 0, 3)] */
v_mul_f32 v[vgprValuC+21], s[sgprAlpha], v[vgprValuC+21] // *= alpha
v_mul_f32 v[vgprValuC+22], s[sgprAlpha], v[vgprValuC+22] // *= alpha
v_mul_f32 v[vgprValuC+23], s[sgprAlpha], v[vgprValuC+23] // *= alpha
v_mul_f32 v[vgprValuC+24], s[sgprAlpha], v[vgprValuC+24] // *= alpha
v_mul_f32 v[vgprValuC+25], s[sgprAlpha], v[vgprValuC+25] // *= alpha
v_mul_f32 v[vgprValuC+26], s[sgprAlpha], v[vgprValuC+26] // *= alpha
v_mul_f32 v[vgprValuC+27], s[sgprAlpha], v[vgprValuC+27] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec (before atomic)
v_add_f32 v38, v39, v[vgprValuC+21]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[62:63]                           // sgprs -> exec (before atomic)
v_add_f32 v43, v44, v[vgprValuC+22]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[64:65]                           // sgprs -> exec (before atomic)
v_add_f32 v46, v47, v[vgprValuC+23]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[66:67]                           // sgprs -> exec (before atomic)
v_add_f32 v49, v50, v[vgprValuC+24]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[68:69]                           // sgprs -> exec (before atomic)
v_add_f32 v52, v53, v[vgprValuC+25]                // desired value avi=0
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[70:71]                           // sgprs -> exec (before atomic)
v_add_f32 v55, v56, v[vgprValuC+26]                // desired value avi=0
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[72:73]                           // sgprs -> exec (before atomic)
v_add_f32 v58, v59, v[vgprValuC+27]                // desired value avi=0
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v38, v39                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v43, v44                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v46, v47                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v49, v50                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v52, v53                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v55, v56                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v58, v59                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0069                         // if exec is zero skip loop

/* atomic CAS loop */
label_0068:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+21]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+22]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+23]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+24]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[68:69]                           // must try again
v_mov_b32 v53, v52                                 // dataV+1 = tmp (new original C)
v_add_f32 v52, v53, v[vgprValuC+25]                // newC = rC + originalC
buffer_atomic_cmpswap v[52:53], v51, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[70:71]                           // must try again
v_mov_b32 v56, v55                                 // dataV+1 = tmp (new original C)
v_add_f32 v55, v56, v[vgprValuC+26]                // newC = rC + originalC
buffer_atomic_cmpswap v[55:56], v54, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[72:73]                           // must try again
v_mov_b32 v59, v58                                 // dataV+1 = tmp (new original C)
v_add_f32 v58, v59, v[vgprValuC+27]                // newC = rC + originalC
buffer_atomic_cmpswap v[58:59], v57, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again
s_mov_b64 exec, s[68:69]                           // must try again
v_cmp_ne_u32 s[54:55], v53, v52                    // c read during atomic == c read during prior load
s_and_b64 s[68:69], s[54:55], s[68:69]             // inBounds & must try again
s_mov_b64 exec, s[70:71]                           // must try again
v_cmp_ne_u32 s[54:55], v56, v55                    // c read during atomic == c read during prior load
s_and_b64 s[70:71], s[54:55], s[70:71]             // inBounds & must try again
s_mov_b64 exec, s[72:73]                           // must try again
v_cmp_ne_u32 s[54:55], v59, v58                    // c read during atomic == c read during prior load
s_and_b64 s[72:73], s[54:55], s[72:73]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[68:69], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[70:71], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[72:73], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0068                        // try again if not complete
label_0069:
s_mov_b64 exec, -1                                 // full mask -> exec
/* optSingleColVgpr=0 optSharedColVgpr=0 optSharedMask=0 optSrdIncForRow=0 */

/******************************************/
/* Global Write Edge Batch #4 (d1,d0,vc1,vc0) = */
/*    (0,7,0,0:vw1:vaw:1); (0,7,0,1:vw1:vaw:1); (0,7,0,2:vw1:vaw:1); (0,7,0,3:vw1:vaw:1) */
/******************************************/

/* calc coords, apply mask, and issue loads (if necessary) */
/* (d1,vc1,d0,vc0)=(0,0,7,0) */
_v_add_co_u32 v35, vcc, v32, 56                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[60:61], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[60:61], s[54:55], s[60:61]             // in0 && in1
_v_add_lshl_u32 v37, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v37, -1, v37, s[60:61]               // clip if OOB. offset
buffer_load_dword v39, v37, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,7,1) */
_v_add_co_u32 v35, vcc, v32, 57                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[62:63], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[62:63], s[54:55], s[62:63]             // in0 && in1
_v_add_lshl_u32 v42, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v42, -1, v42, s[62:63]               // clip if OOB. offset
buffer_load_dword v44, v42, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,7,2) */
_v_add_co_u32 v35, vcc, v32, 58                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[64:65], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[64:65], s[54:55], s[64:65]             // in0 && in1
_v_add_lshl_u32 v45, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v45, -1, v45, s[64:65]               // clip if OOB. offset
buffer_load_dword v47, v45, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1
/* (d1,vc1,d0,vc0)=(0,0,7,3) */
_v_add_co_u32 v35, vcc, v32, 59                    // coord0.1: coord0 += d0*sg0*VW + vc0
v_cmp_lt_u32 s[54:55], v35, s[sgprSizeI]           // coord0 < size0
v_cmp_lt_u32 s[66:67], v33, s[sgprSizeJ]           // coord1 < size1
s_and_b64 s[66:67], s[54:55], s[66:67]             // in0 && in1
_v_add_lshl_u32 v48, v34, v35, 0x2                 // scaleToBpe: accumulate d0 lower and *= bpe into Cin addr
v_cndmask_b32 v48, -1, v48, s[66:67]               // clip if OOB. offset
buffer_load_dword v50, v48, s[sgprSrdC:sgprSrdC+3], 0, offen offset:0 // load C (atomic) bpm=4 vaw=1

/* rC *= alpha batchEements=[(0, 7, 0, 0), (0, 7, 0, 1), (0, 7, 0, 2), (0, 7, 0, 3)] */
v_mul_f32 v[vgprValuC+28], s[sgprAlpha], v[vgprValuC+28] // *= alpha
v_mul_f32 v[vgprValuC+29], s[sgprAlpha], v[vgprValuC+29] // *= alpha
v_mul_f32 v[vgprValuC+30], s[sgprAlpha], v[vgprValuC+30] // *= alpha
v_mul_f32 v[vgprValuC+31], s[sgprAlpha], v[vgprValuC+31] // *= alpha
s_waitcnt vmcnt(0)                                 // wait C (atomic)

/* issue first atomic writes */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec (before atomic)
v_add_f32 v38, v39, v[vgprValuC+28]                // desired value avi=0
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[62:63]                           // sgprs -> exec (before atomic)
v_add_f32 v43, v44, v[vgprValuC+29]                // desired value avi=0
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[64:65]                           // sgprs -> exec (before atomic)
v_add_f32 v46, v47, v[vgprValuC+30]                // desired value avi=0
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_mov_b64 exec, s[66:67]                           // sgprs -> exec (before atomic)
v_add_f32 v49, v50, v[vgprValuC+31]                // desired value avi=0
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // attempt write avi=0
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* check success of writes, update masks */
s_mov_b64 exec, s[60:61]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v38, v39                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v43, v44                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v46, v47                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // sgprs -> exec
v_cmp_ne_u32 s[54:55], v49, v50                    // c read during atomic == c read during prior load (avi=0, first)
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execz label_0071                         // if exec is zero skip loop

/* atomic CAS loop */
label_0070:

/* apply updated masks and issue writes again */
s_mov_b64 exec, s[60:61]                           // must try again
v_mov_b32 v39, v38                                 // dataV+1 = tmp (new original C)
v_add_f32 v38, v39, v[vgprValuC+28]                // newC = rC + originalC
buffer_atomic_cmpswap v[38:39], v37, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[62:63]                           // must try again
v_mov_b32 v44, v43                                 // dataV+1 = tmp (new original C)
v_add_f32 v43, v44, v[vgprValuC+29]                // newC = rC + originalC
buffer_atomic_cmpswap v[43:44], v42, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[64:65]                           // must try again
v_mov_b32 v47, v46                                 // dataV+1 = tmp (new original C)
v_add_f32 v46, v47, v[vgprValuC+30]                // newC = rC + originalC
buffer_atomic_cmpswap v[46:47], v45, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_mov_b64 exec, s[66:67]                           // must try again
v_mov_b32 v50, v49                                 // dataV+1 = tmp (new original C)
v_add_f32 v49, v50, v[vgprValuC+31]                // newC = rC + originalC
buffer_atomic_cmpswap v[49:50], v48, s[sgprSrdD:sgprSrdD+3] 0 offen offset:0 glc    // try again
s_waitcnt vmcnt(0)                                 // wait for atomic writes

/* apply masks and check for success */
s_mov_b64 exec, s[60:61]                           // must try again
v_cmp_ne_u32 s[54:55], v39, v38                    // c read during atomic == c read during prior load
s_and_b64 s[60:61], s[54:55], s[60:61]             // inBounds & must try again
s_mov_b64 exec, s[62:63]                           // must try again
v_cmp_ne_u32 s[54:55], v44, v43                    // c read during atomic == c read during prior load
s_and_b64 s[62:63], s[54:55], s[62:63]             // inBounds & must try again
s_mov_b64 exec, s[64:65]                           // must try again
v_cmp_ne_u32 s[54:55], v47, v46                    // c read during atomic == c read during prior load
s_and_b64 s[64:65], s[54:55], s[64:65]             // inBounds & must try again
s_mov_b64 exec, s[66:67]                           // must try again
v_cmp_ne_u32 s[54:55], v50, v49                    // c read during atomic == c read during prior load
s_and_b64 s[66:67], s[54:55], s[66:67]             // inBounds & must try again

/* or masks to check for exit */
s_mov_b64 s[54:55], 0x0                            // empty mask
s_or_b64 s[54:55], s[60:61], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[62:63], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[64:65], s[54:55]              // or to add threads
s_or_b64 s[54:55], s[66:67], s[54:55]              // or to add threads
s_or_saveexec_b64 s[56:57], s[54:55]               // apply combined mask
s_cbranch_execnz label_0070                        // try again if not complete
label_0071:
s_mov_b64 exec, -1                                 // full mask -> exec
s_branch label_0023                                // jump to end
label_0023:

label_0072:  /// KernelEnd
s_endpgm                                           // Kernel End


