

/******************************************/
/* Function Prefix                        */
/******************************************/



/******************************************/
/* Begin Kernel                           */
/******************************************/

.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 8, "AMD", "AMDGPU" 
.text
.protected Cijk_Alik_Bljk_SB_MT64x128x32_SE_K1
.globl Cijk_Alik_Bljk_SB_MT64x128x32_SE_K1
.p2align 8
.type Cijk_Alik_Bljk_SB_MT64x128x32_SE_K1,@function
.amdgpu_hsa_kernel Cijk_Alik_Bljk_SB_MT64x128x32_SE_K1
Cijk_Alik_Bljk_SB_MT64x128x32_SE_K1:
.amd_kernel_code_t
  is_ptr64 = 1
  enable_sgpr_kernarg_segment_ptr = 1
  kernarg_segment_byte_size = 148 // bytes of kern args
  workitem_vgpr_count = 108 // vgprs
  wavefront_sgpr_count = 98 // sgprs
  compute_pgm_rsrc1_vgprs = 26 // floor((108-1)/4)
  compute_pgm_rsrc1_sgprs = 12 // floor((98-1)/8)
  compute_pgm_rsrc2_tidig_comp_cnt = 0 // 1D wg
  compute_pgm_rsrc2_tgid_x_en = 1 // wg.x
  compute_pgm_rsrc2_tgid_y_en = 1 // wg.y
  compute_pgm_rsrc2_tgid_z_en = 1 // wg.z
  workgroup_group_segment_byte_size = 60000// lds bytes
  compute_pgm_rsrc2_user_sgpr = 2 // vcc
  kernarg_segment_alignment = 4
  group_segment_alignment = 4
  private_segment_alignment = 4
.end_amd_kernel_code_t

/******************************************/
/* Optimizations and Config:              */
/******************************************/
/* ThreadTile= 4 x 8 */
/* SubGroup= 16 x 16 */
/* VectorWidth=4 */
/* GlobalLoadVectorWidthA=4, GlobalLoadVectorWidthB=4 */
/* DirectToLdsA=False */
/* DirectToLdsB=False */
/* UseSgprForGRO=1 */
.amd_amdgpu_hsa_metadata
Version: [ 1, 0 ]
Kernels:
  - Name: Cijk_Alik_Bljk_SB_MT64x128x32_SE_K1
    SymbolName: 'Cijk_Alik_Bljk_SB_MT64x128x32_SE_K1@kd'
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
      - Name:            beta
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       F32
      - Name:            strideD0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            strideD1
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            strideC0
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            strideC1
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
      - Name:            SizesFree2
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       U32
      - Name:            SizesSum0
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
      KernargSegmentSize: 148
      GroupSegmentFixedSize: 57344
      PrivateSegmentFixedSize: 0
      KernargSegmentAlign:  8
      WavefrontSize:        64
      NumSGPRs:             98
      NumVGPRs:             108
      MaxFlatWorkGroupSize: 256
.end_amd_amdgpu_hsa_metadata

/******************************************/
/* Asm syntax workarounds                 */
/******************************************/
.macro _v_add_co_u32 dst, cc, src0, src1, dpp=
   v_add_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_add_u32 dst, src0, src1, dpp=
   v_add_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_sub_co_u32 dst, cc, src0, src1, dpp=
   v_sub_co_u32 \dst, \cc, \src0, \src1 \dpp
.endm

.macro _v_sub_u32 dst, src0, src1, dpp=
   v_sub_u32 \dst, \src0, \src1 \dpp
.endm

.macro _v_addc_co_u32 dst, ccOut, src0, ccIn, src1, dpp=
   v_addc_co_u32 \dst, \ccOut, \src0, \ccIn, \src1 \dpp
.endm

.macro _v_add_lshl_u32 dst, src0, src1, shiftCnt
    v_add_lshl_u32 \dst, \src0, \src1, \shiftCnt
.endm

.macro _v_lshl_add_u32 dst, src0, src1, shiftCnt
    v_lshl_add_u32 \dst, \src0, \src1, \shiftCnt
.endm

/******************************************/
/* Magic div and mod functions            */
/******************************************/
.macro V_MAGIC_DIV dstIdx, dividend, magicNumber, magicShift
    v_mul_hi_u32 v[\dstIdx+1], \dividend, \magicNumber
    v_mul_lo_u32 v[\dstIdx+0], \dividend, \magicNumber
    v_lshrrev_b64 v[\dstIdx:\dstIdx+1], \magicShift, v[\dstIdx:\dstIdx+1]
.endm

/******************************************/
/* VGPR Assignments                       */
/******************************************/
.set vgprValuC, 0
/* ValuA/B   Xn=PLR buffer idx,  In=InnerUnroll idx */
.set vgprValuA_X0_I0, 32
.set vgprValuA_X1_I0, 36
.set vgprG2LA, 40
.set vgprValuB_X0_I0, 48
.set vgprValuB_X1_I0, 56
.set vgprG2LB, 64
.set vgprLocalWriteAddrA, 80
.set vgprLocalWriteAddrB, 81
.set vgprGlobalReadOffsetA, 82
.set vgprGlobalReadOffsetB, 83
.set vgprLocalReadAddrA, 84
.set vgprLocalReadAddrB, 85
.set vgprSerial, 86
/* Num VGPR=87 */

/******************************************/
/* SGPR Assignments                       */
/******************************************/
.set sgprKernArgAddress, 0
.set sgprWorkGroup0, 2
.set sgprWorkGroup1, 3
.set sgprWorkGroup2, 4
.set sgprNumWorkGroups0, 5
.set sgprNumWorkGroups1, 6
.set sgprSrdA, 8
.set sgprSrdB, 12
.set sgprSrdD, 16
.set sgprSrdC, 20
.set sgprTensor2dSizeC, 24
.set sgprTensor2dSizeA, 26
.set sgprTensor2dSizeB, 28
.set sgprSaveExecMask, 30
.set sgprAddressD, 32
.set sgprAddressC, 34
.set sgprStridesD, 36
.set sgprStridesC, 38
.set sgprAlpha, 40
.set sgprBeta, 41
.set sgprSizesFree, 42
.set sgprSizesSum, 45
.set sgprLoopCounters, 46
.set sgprOrigLoopCounter, 47
.set sgprStridesA, 48
.set sgprStridesB, 50
.set sgprAddressA, 52
.set sgprAddressB, 54
.set sgprShadowLimitA, 56
.set sgprShadowLimitB, 58
.set sgprOrigStaggerUIter, 60
.set sgprStaggerUIter, 61
.set sgprWrapUA, 62
.set sgprWrapUB, 64
.set sgprNumFullBlocks, 66
.set sgprWgmRemainder1, 67
.set sgprMagicNumberWgmRemainder1, 68
.set sgprGlobalReadIncsA, 69
.set sgprGlobalReadIncsB, 70
.set sgprScalarGlobalReadOffsetA, 71
.set sgprScalarGlobalReadOffsetB, 72
/* max SGPR=98 */

/* Size Assignments */
.set sgprSizeD0I, sgprSizesFree+0
.set sgprSizeD1J, sgprSizesFree+1
.set sgprSizeDK, sgprSizesFree+2
.set sgprSizeC0I, sgprSizesFree+0
.set sgprSizeC1J, sgprSizesFree+1
.set sgprSizeCK, sgprSizesFree+2
.set sgprSizeAL, sgprSizesSum+0
.set sgprSizeA0I, sgprSizesFree+0
.set sgprSizeAK, sgprSizesFree+2
.set sgprSizeBL, sgprSizesSum+0
.set sgprSizeB1J, sgprSizesFree+1
.set sgprSizeBK, sgprSizesFree+2

/* Stride Assignments */
.set constStrideD0I, 1
.set sgprStrideD1J, sgprStridesD+0
.set sgprStrideDK, sgprStridesD+1
.set constStrideC0I, 1
.set sgprStrideC1J, sgprStridesC+0
.set sgprStrideCK, sgprStridesC+1
.set constStrideAL, 1
.set sgprStrideA0I, sgprStridesA+0
.set sgprStrideAK, sgprStridesA+1
.set constStrideBL, 1
.set sgprStrideB1J, sgprStridesB+0
.set sgprStrideBK, sgprStridesB+1

.set DepthU, 32
/* Number of elements to shift-left SRD */
.set SrdShiftLeftA, 4
.set SrdShiftLeftB, 4
/* 2GB limit - set offsets to -1 to exceed this and clamp */
.set BufferLimit, 0x80000000
/* Bits 127:96 of SRD.  Set DataFormat = 32 bit */
.set Srd127_96, 0x0020000
.set BufferOOB, 0x80000000











.long 0xC00A0D00, 0x00000028
.long 0xC00A0C00, 0x00000050
.long 0xC00A0600, 0x00000008
.long 0xC0020B40, 0x0000006C
.long 0xBEFC00FF, 0x00006000
.long 0x7EC80300
.long 0x26CA00BF
.long 0x2004C886
.long 0xB8D0F804
.long 0xD1130004, 0x0000A0B0
.long 0x20CC0884
.long 0x7EA40566
.long 0xD1130067, 0x0000A08F
.long 0x7EA20567
.long 0xBF068151
.long 0xBF84020E
.long 0xBF8CC07F
.long 0xBE880034
.long 0xBE890035
.long 0xBE8B00FF, 0x00020000
.long 0x80B85418
.long 0x80B95518
.long 0x8EB88238
.long 0x80388438
.long 0x82398039
.long 0xBF068039
.long 0x850AFF38, 0x80000000
.long 0xBE8A00FF, 0x80000000
.long 0x9254C030
.long 0x92545402
.long 0x8E558452
.long 0x92533055
.long 0x81545354
.long 0x2000CA85
.long 0xD2850004, 0x00020030
.long 0x2602CA9F
.long 0x32A40304
.long 0x68A4A454
.long 0x24A4A482
.long 0x8E478330
.long 0x80C7FF47, 0x00000108
.long 0x68A6A447
.long 0x68A8A647
.long 0x68AAA847
.long 0x68ACAA47
.long 0x68AEAC47
.long 0x68B0AE47
.long 0x68B2B047
.long 0xBECC00FF, 0x00000840
.long 0x924C4C52
.long 0xBE8C0036
.long 0xBE8D0037
.long 0xBE8F00FF, 0x00020000
.long 0x80BA541A
.long 0x80BB551A
.long 0x8EBA823A
.long 0x803A843A
.long 0x823B803B
.long 0xBF06803B
.long 0x850EFF3A, 0x80000000
.long 0xBE8E00FF, 0x80000000
.long 0x9254FF32, 0x00000080
.long 0x92545403
.long 0x925532A0
.long 0x92555552
.long 0x81545554
.long 0x2004CA85
.long 0x2606CA9F
.long 0xD2850004, 0x00020432
.long 0x32400704
.long 0x68404054
.long 0x24404082
.long 0x8E4A8332
.long 0x80CAFF4A, 0x00000108
.long 0x6842404A
.long 0x6844424A
.long 0x6846444A
.long 0x6848464A
.long 0x684A484A
.long 0x684C4A4A
.long 0x684E4C4A
.long 0x68504E4A
.long 0x6852504A
.long 0x6854524A
.long 0x6856544A
.long 0x6858564A
.long 0x685A584A
.long 0x685C5A4A
.long 0x685E5C4A
.long 0xBECE00FF, 0x00001080
.long 0x924E4E52
.long 0x814EFF4E, 0x00004200
.long 0xBF8A0000
.long 0xBEFC004C
.long 0x814DFF4C, 0x00002100
.long 0xE0511000, 0x80023052
.long 0xE0511108, 0x80023153
.long 0xE0511210, 0x80023254
.long 0xE0511318, 0x80023355
.long 0xE0511420, 0x80023456
.long 0xE0511528, 0x80023557
.long 0xE0511630, 0x80023658
.long 0xE0511738, 0x80023759
.long 0xBEFC004E
.long 0x814FFF4E, 0x00004200
.long 0xE0511000, 0x80033020
.long 0xE0511108, 0x80033121
.long 0xE0511210, 0x80033222
.long 0xE0511318, 0x80033323
.long 0xE0511420, 0x80033424
.long 0xE0511528, 0x80033525
.long 0xE0511630, 0x80033626
.long 0xE0511738, 0x80033727
.long 0xE0511840, 0x80033828
.long 0xE0511948, 0x80033929
.long 0xE0511A50, 0x80033A2A
.long 0xE0511B58, 0x80033B2B
.long 0xE0511C60, 0x80033C2C
.long 0xE0511D68, 0x80033D2D
.long 0xE0511E70, 0x80033E2E
.long 0xE0511F78, 0x80033F2F
.long 0xBEFC004D
.long 0x68A4A4FF, 0x00000080
.long 0x68A6A6FF, 0x00000080
.long 0x68A8A8FF, 0x00000080
.long 0x68AAAAFF, 0x00000080
.long 0x68ACACFF, 0x00000080
.long 0x68AEAEFF, 0x00000080
.long 0x68B0B0FF, 0x00000080
.long 0x68B2B2FF, 0x00000080
.long 0x684040FF, 0x00000080
.long 0x684242FF, 0x00000080
.long 0x684444FF, 0x00000080
.long 0x684646FF, 0x00000080
.long 0x684848FF, 0x00000080
.long 0x684A4AFF, 0x00000080
.long 0x684C4CFF, 0x00000080
.long 0x684E4EFF, 0x00000080
.long 0x685050FF, 0x00000080
.long 0x685252FF, 0x00000080
.long 0x685454FF, 0x00000080
.long 0x685656FF, 0x00000080
.long 0x685858FF, 0x00000080
.long 0x685A5AFF, 0x00000080
.long 0x685C5CFF, 0x00000080
.long 0x685E5EFF, 0x00000080
.long 0xE0511000, 0x80023052
.long 0xE0511108, 0x80023153
.long 0xE0511210, 0x80023254
.long 0xE0511318, 0x80023355
.long 0xE0511420, 0x80023456
.long 0xE0511528, 0x80023557
.long 0xE0511630, 0x80023658
.long 0xE0511738, 0x80023759
.long 0xBEFC004F
.long 0xBF800000
.long 0xE0511000, 0x80033020
.long 0xE0511108, 0x80033121
.long 0xE0511210, 0x80033222
.long 0xE0511318, 0x80033323
.long 0xE0511420, 0x80033424
.long 0xE0511528, 0x80033525
.long 0xE0511630, 0x80033626
.long 0xE0511738, 0x80033727
.long 0xE0511840, 0x80033828
.long 0xE0511948, 0x80033929
.long 0xE0511A50, 0x80033A2A
.long 0xE0511B58, 0x80033B2B
.long 0xE0511C60, 0x80033C2C
.long 0xE0511D68, 0x80033D2D
.long 0xE0511E70, 0x80033E2E
.long 0xE0511F78, 0x80033F2F
.long 0x68A4A4FF, 0x00000080
.long 0x68A6A6FF, 0x00000080
.long 0x68A8A8FF, 0x00000080
.long 0x68AAAAFF, 0x00000080
.long 0x68ACACFF, 0x00000080
.long 0x68AEAEFF, 0x00000080
.long 0x68B0B0FF, 0x00000080
.long 0x68B2B2FF, 0x00000080
.long 0x684040FF, 0x00000080
.long 0x684242FF, 0x00000080
.long 0x684444FF, 0x00000080
.long 0x684646FF, 0x00000080
.long 0x684848FF, 0x00000080
.long 0x684A4AFF, 0x00000080
.long 0x684C4CFF, 0x00000080
.long 0x684E4EFF, 0x00000080
.long 0x685050FF, 0x00000080
.long 0x685252FF, 0x00000080
.long 0x685454FF, 0x00000080
.long 0x685656FF, 0x00000080
.long 0x685858FF, 0x00000080
.long 0x685A5AFF, 0x00000080
.long 0x685C5CFF, 0x00000080
.long 0x685E5EFF, 0x00000080
.long 0xBEFC004C
.long 0xBF8C8F78
.long 0xBF8A0000
.long 0xBF8C4F78
.long 0xBF8A0000
.long 0x8F2E852D
.long 0x80AE2E80
.long 0xBF06802E
.long 0xBF8500DC
.long 0xBF8A0000
.long 0xE0511000, 0x80023052
.long 0xE0511108, 0x80023153
.long 0xE0511210, 0x80023254
.long 0xE0511318, 0x80023355
.long 0xE0511420, 0x80023456
.long 0xE0511528, 0x80023557
.long 0xE0511630, 0x80023658
.long 0xE0511738, 0x80023759
.long 0xBEFC004E
.long 0xBF800000
.long 0xE0511000, 0x80033020
.long 0xE0511108, 0x80033121
.long 0xE0511210, 0x80033222
.long 0xE0511318, 0x80033323
.long 0xE0511420, 0x80033424
.long 0xE0511528, 0x80033525
.long 0xE0511630, 0x80033626
.long 0xE0511738, 0x80033727
.long 0xE0511840, 0x80033828
.long 0xE0511948, 0x80033929
.long 0xE0511A50, 0x80033A2A
.long 0xE0511B58, 0x80033B2B
.long 0xE0511C60, 0x80033C2C
.long 0xE0511D68, 0x80033D2D
.long 0xE0511E70, 0x80033E2E
.long 0xE0511F78, 0x80033F2F
.long 0xBF8C8F78
.long 0xBF8F0001
.long 0xBF8A0000
.long 0x68A4A4FF, 0x00000080
.long 0x68A6A6FF, 0x00000080
.long 0x68A8A8FF, 0x00000080
.long 0x68AAAAFF, 0x00000080
.long 0x68ACACFF, 0x00000080
.long 0x68AEAEFF, 0x00000080
.long 0x68B0B0FF, 0x00000080
.long 0x68B2B2FF, 0x00000080
.long 0x684040FF, 0x00000080
.long 0x684242FF, 0x00000080
.long 0x684444FF, 0x00000080
.long 0x684646FF, 0x00000080
.long 0x684848FF, 0x00000080
.long 0x684A4AFF, 0x00000080
.long 0x684C4CFF, 0x00000080
.long 0x684E4EFF, 0x00000080
.long 0xBF8F0000
.long 0xBF8C4F78
.long 0xBF8F0001
.long 0xBF8A0000
.long 0x685050FF, 0x00000080
.long 0x685252FF, 0x00000080
.long 0x685454FF, 0x00000080
.long 0x685656FF, 0x00000080
.long 0x685858FF, 0x00000080
.long 0x685A5AFF, 0x00000080
.long 0x685C5CFF, 0x00000080
.long 0x685E5EFF, 0x00000080
.long 0xBF8F0000
.long 0xBEFC004D
.long 0x802E812E
.long 0xBF8A0000
.long 0xE0511000, 0x80023052
.long 0xE0511108, 0x80023153
.long 0xE0511210, 0x80023254
.long 0xE0511318, 0x80023355
.long 0xE0511420, 0x80023456
.long 0xE0511528, 0x80023557
.long 0xE0511630, 0x80023658
.long 0xE0511738, 0x80023759
.long 0xBEFC004F
.long 0xBF800000
.long 0xE0511000, 0x80033020
.long 0xE0511108, 0x80033121
.long 0xE0511210, 0x80033222
.long 0xE0511318, 0x80033323
.long 0xE0511420, 0x80033424
.long 0xE0511528, 0x80033525
.long 0xE0511630, 0x80033626
.long 0xE0511738, 0x80033727
.long 0xE0511840, 0x80033828
.long 0xE0511948, 0x80033929
.long 0xE0511A50, 0x80033A2A
.long 0xE0511B58, 0x80033B2B
.long 0xE0511C60, 0x80033C2C
.long 0xE0511D68, 0x80033D2D
.long 0xE0511E70, 0x80033E2E
.long 0xE0511F78, 0x80033F2F
.long 0xBF8C8F78
.long 0xBF8A0000
.long 0xBF8F0001
.long 0x68A4A4FF, 0x00000080
.long 0x68A6A6FF, 0x00000080
.long 0x68A8A8FF, 0x00000080
.long 0x68AAAAFF, 0x00000080
.long 0x68ACACFF, 0x00000080
.long 0x68AEAEFF, 0x00000080
.long 0x68B0B0FF, 0x00000080
.long 0x68B2B2FF, 0x00000080
.long 0x684040FF, 0x00000080
.long 0x684242FF, 0x00000080
.long 0x684444FF, 0x00000080
.long 0x684646FF, 0x00000080
.long 0x684848FF, 0x00000080
.long 0x684A4AFF, 0x00000080
.long 0x684C4CFF, 0x00000080
.long 0x684E4EFF, 0x00000080
.long 0xBF8F0000
.long 0xBF8C4F78
.long 0xBF8A0000
.long 0xBF8F0001
.long 0x685050FF, 0x00000080
.long 0x685252FF, 0x00000080
.long 0x685454FF, 0x00000080
.long 0x685656FF, 0x00000080
.long 0x685858FF, 0x00000080
.long 0x685A5AFF, 0x00000080
.long 0x685C5CFF, 0x00000080
.long 0x685E5EFF, 0x00000080
.long 0xBF8F0000
.long 0xBEFC004C
.long 0x802E812E
.long 0xBF00C22E
.long 0xBF84FF24
.long 0xBF8C4F70
.long 0xBF8A0000
.long 0xBF8C0F70
.long 0xBF8A0000
.long 0xBF810000
.long 0xD3D94000, 0x18000080
.long 0xD3D94001, 0x18000080
.long 0xD3D94002, 0x18000080
.long 0xD3D94003, 0x18000080
.long 0xD3D94004, 0x18000080
.long 0xD3D94005, 0x18000080
.long 0xD3D94006, 0x18000080
.long 0xD3D94007, 0x18000080
.long 0xD3D94008, 0x18000080
.long 0xD3D94009, 0x18000080
.long 0xD3D9400A, 0x18000080
.long 0xD3D9400B, 0x18000080
.long 0xD3D9400C, 0x18000080
.long 0xD3D9400D, 0x18000080
.long 0xD3D9400E, 0x18000080
.long 0xD3D9400F, 0x18000080
.long 0xD3D94010, 0x18000080
.long 0xD3D94011, 0x18000080
.long 0xD3D94012, 0x18000080
.long 0xD3D94013, 0x18000080
.long 0xD3D94014, 0x18000080
.long 0xD3D94015, 0x18000080
.long 0xD3D94016, 0x18000080
.long 0xD3D94017, 0x18000080
.long 0xD3D94018, 0x18000080
.long 0xD3D94019, 0x18000080
.long 0xD3D9401A, 0x18000080
.long 0xD3D9401B, 0x18000080
.long 0xD3D9401C, 0x18000080
.long 0xD3D9401D, 0x18000080
.long 0xD3D9401E, 0x18000080
.long 0xD3D9401F, 0x18000080
.long 0xC0060700, 0x00000000
.long 0xC00A0A00, 0x00000038
.long 0xC00A0900, 0x00000040
.long 0xC00A0800, 0x00000018
.long 0xD1130001, 0x00013F65
.long 0xD2850060, 0x000202A0
.long 0x20020281
.long 0xD2850001, 0x00020282
.long 0x68C0C101
.long 0x2002CA85
.long 0x68C0C101
.long 0x24C0C082
.long 0x68C0C080
.long 0x68C2C0FF, 0x00002100
.long 0xBF8A0000
.long 0xD1130001, 0x00013F65
.long 0xD2850062, 0x000202A0
.long 0x20020281
.long 0xD2850001, 0x00020282
.long 0x68C4C501
.long 0x2002CA85
.long 0x68C4C501
.long 0x24C4C482
.long 0x9254FF52, 0x00001080
.long 0x68C4C454
.long 0x68C4C4FF, 0x00004200
.long 0x68C6C4FF, 0x00004200
.long 0xBF8CC07F
.long 0xBE900022
.long 0xBE910023
.long 0xBE9200FF, 0x80000000
.long 0xBE9300FF, 0x00020000
.long 0xBE940020
.long 0xBE950021
.long 0xBE9600FF, 0x80000000
.long 0xBE9700FF, 0x00020000
.long 0x925603FF, 0x00000080
.long 0x96552656
.long 0x92542656
.long 0x8ED48254
.long 0x80105410
.long 0x82115511
.long 0x80145414
.long 0x82155515
.long 0x96552704
.long 0x92542704
.long 0x8ED48254
.long 0x80105410
.long 0x82115511
.long 0x80145414
.long 0x82155515
.long 0x24C8CC86
.long 0x68C8C965
.long 0xD2850004, 0x0002CCA0
.long 0xD2850003, 0x00004D04
.long 0x2608C89F
.long 0xD2850005, 0x00004D04
.long 0x2608C8BF
.long 0x200C0885
.long 0x240C0C82
.long 0x68D60B03
.long 0x925402C0
.long 0x32D40C54
.long 0xD1FE0068, 0x020AD76A
.long 0xBF8A0000
.long 0xD86C0000, 0x20000060
.long 0xD86C1080, 0x21000060
.long 0xD86C0008, 0x22000060
.long 0xD86C1088, 0x23000060
.long 0xD86C0010, 0x24000060
.long 0xD86C1090, 0x25000060
.long 0xD86C0018, 0x26000060
.long 0xD86C1098, 0x27000060
.long 0xD86C0020, 0x28000060
.long 0xD86C10A0, 0x29000060
.long 0xD86C0028, 0x2A000060
.long 0xD86C10A8, 0x2B000060
.long 0xD86C0030, 0x2C000060
.long 0xD86C10B0, 0x2D000060
.long 0xD86C0038, 0x2E000060
.long 0xD86C10B8, 0x2F000060
.long 0xD86C0040, 0x30000060
.long 0xD86C10C0, 0x31000060
.long 0xD86C0048, 0x32000060
.long 0xD86C10C8, 0x33000060
.long 0xD86C0050, 0x34000060
.long 0xD86C10D0, 0x35000060
.long 0xD86C0058, 0x36000060
.long 0xD86C10D8, 0x37000060
.long 0xD86C0060, 0x38000060
.long 0xD86C10E0, 0x39000060
.long 0xD86C0068, 0x3A000060
.long 0xD86C10E8, 0x3B000060
.long 0xD86C0070, 0x3C000060
.long 0xD86C10F0, 0x3D000060
.long 0xD86C0078, 0x3E000060
.long 0xD86C10F8, 0x3F000060
.long 0xBF8A0000
.long 0xD86C0000, 0x00000062
.long 0xD86C0008, 0x01000062
.long 0xD86C0010, 0x02000062
.long 0xD86C0018, 0x03000062
.long 0xD86C0020, 0x04000062
.long 0xD86C0028, 0x05000062
.long 0x8F2E852D
.long 0x80AE2E80
.long 0xBF06802E
.long 0xBF85048A
.long 0xBF8CC47F
.long 0xD3C40000, 0x04020120
.long 0xD86C0030, 0x06000062
.long 0xD86C0038, 0x07000062
.long 0xD3C40010, 0x04420121
.long 0xD86C0040, 0x08000062
.long 0xD86C0048, 0x09000062
.long 0xD3C40000, 0x04020322
.long 0xD86C0050, 0x0A000062
.long 0xD86C0058, 0x0B000062
.long 0xD86C0060, 0x0C000062
.long 0xD3C40010, 0x04420323
.long 0xD86C0068, 0x0D000062
.long 0xD86C0070, 0x0E000062
.long 0xD86C0078, 0x0F000062
.long 0xBF8A0000
.long 0xBF8CCD7F
.long 0xD3C40000, 0x04020524
.long 0xD3C40010, 0x04420525
.long 0xBF8CCC7F
.long 0xD3C40000, 0x04020726
.long 0xD3C40010, 0x04420727
.long 0xBF8CCB7F
.long 0xD3C40000, 0x04020928
.long 0xD3C40010, 0x04420929
.long 0xBF8CCA7F
.long 0xD3C40000, 0x04020B2A
.long 0xD3C40010, 0x04420B2B
.long 0xBF8CC97F
.long 0xD3C40000, 0x04020D2C
.long 0xD3C40010, 0x04420D2D
.long 0xBF8CC87F
.long 0xD3C40000, 0x04020F2E
.long 0xD3C40010, 0x04420F2F
.long 0xBF8CC07F
.long 0xD3C40000, 0x04021130
.long 0xD3C40010, 0x04421131
.long 0xD3C40000, 0x04021332
.long 0xD3C40010, 0x04421333
.long 0xBF8F0000
.long 0xD3C40000, 0x04021534
.long 0xD3C40010, 0x04421535
.long 0xBF8A0000
.long 0xD3C40000, 0x04021736
.long 0xD86C0000, 0x40000061
.long 0xD86C1080, 0x41000061
.long 0xD86C0008, 0x42000061
.long 0xD86C1088, 0x43000061
.long 0xD3C40010, 0x04421737
.long 0xD86C0010, 0x44000061
.long 0xD86C1090, 0x45000061
.long 0xD86C0018, 0x46000061
.long 0xD86C1098, 0x47000061
.long 0xD3C40000, 0x04021938
.long 0xD86C0020, 0x48000061
.long 0xD86C10A0, 0x49000061
.long 0xD86C0028, 0x4A000061
.long 0xD86C10A8, 0x4B000061
.long 0xD3C40010, 0x04421939
.long 0xD86C0030, 0x4C000061
.long 0xD86C10B0, 0x4D000061
.long 0xD86C0038, 0x4E000061
.long 0xD86C10B8, 0x4F000061
.long 0xD3C40000, 0x04021B3A
.long 0xD86C0040, 0x50000061
.long 0xD86C10C0, 0x51000061
.long 0xD86C0048, 0x52000061
.long 0xD86C10C8, 0x53000061
.long 0xD3C40010, 0x04421B3B
.long 0xD86C0050, 0x54000061
.long 0xD86C10D0, 0x55000061
.long 0xD86C0058, 0x56000061
.long 0xD86C10D8, 0x57000061
.long 0xD3C40000, 0x04021D3C
.long 0xD86C0060, 0x58000061
.long 0xD86C10E0, 0x59000061
.long 0xD86C0068, 0x5A000061
.long 0xD86C10E8, 0x5B000061
.long 0xD3C40010, 0x04421D3D
.long 0xD86C0070, 0x5C000061
.long 0xD86C10F0, 0x5D000061
.long 0xD86C0078, 0x5E000061
.long 0xD86C10F8, 0x5F000061
.long 0xBF8A0000
.long 0xD86C0000, 0x10000063
.long 0xD86C0008, 0x11000063
.long 0xD3C40000, 0x04021F3E
.long 0xD86C0010, 0x12000063
.long 0xD86C0018, 0x13000063
.long 0xD86C0020, 0x14000063
.long 0xD86C0028, 0x15000063
.long 0xD86C0030, 0x16000063
.long 0xD3C40010, 0x04421F3F
.long 0xD86C0038, 0x17000063
.long 0xD86C0040, 0x18000063
.long 0xD86C0048, 0x19000063
.long 0xD86C0050, 0x1A000063
.long 0xD86C0058, 0x1B000063
.long 0xBF8F0001
.long 0x802E812E
.long 0xBF8CCA7F
.long 0xD3C40000, 0x04022140
.long 0xD86C0060, 0x1C000063
.long 0xD86C0068, 0x1D000063
.long 0xD3C40010, 0x04422141
.long 0xD86C0070, 0x1E000063
.long 0xD86C0078, 0x1F000063
.long 0xD3C40000, 0x04022342
.long 0xD3C40010, 0x04422343
.long 0xBF8A0000
.long 0xBF8CCD7F
.long 0xD3C40000, 0x04022544
.long 0xD3C40010, 0x04422545
.long 0xBF8CCC7F
.long 0xD3C40000, 0x04022746
.long 0xD3C40010, 0x04422747
.long 0xBF8CCB7F
.long 0xD3C40000, 0x04022948
.long 0xD3C40010, 0x04422949
.long 0xBF8CCA7F
.long 0xD3C40000, 0x04022B4A
.long 0xD3C40010, 0x04422B4B
.long 0xBF8CC97F
.long 0xD3C40000, 0x04022D4C
.long 0xD3C40010, 0x04422D4D
.long 0xBF8CC87F
.long 0xD3C40000, 0x04022F4E
.long 0xD3C40010, 0x04422F4F
.long 0xBF8CC07F
.long 0xD3C40000, 0x04023150
.long 0xD3C40010, 0x04423151
.long 0xD3C40000, 0x04023352
.long 0xD3C40010, 0x04423353
.long 0xBF8F0000
.long 0xD3C40000, 0x04023554
.long 0xD3C40010, 0x04423555
.long 0xBF8A0000
.long 0xD86C0000, 0x20000060
.long 0xD86C1080, 0x21000060
.long 0xD86C0008, 0x22000060
.long 0xD86C1088, 0x23000060
.long 0xD3C40000, 0x04023756
.long 0xD86C0010, 0x24000060
.long 0xD86C1090, 0x25000060
.long 0xD86C0018, 0x26000060
.long 0xD86C1098, 0x27000060
.long 0xD3C40010, 0x04423757
.long 0xD86C0020, 0x28000060
.long 0xD86C10A0, 0x29000060
.long 0xD86C0028, 0x2A000060
.long 0xD86C10A8, 0x2B000060
.long 0xD3C40000, 0x04023958
.long 0xD86C0030, 0x2C000060
.long 0xD86C10B0, 0x2D000060
.long 0xD86C0038, 0x2E000060
.long 0xD86C10B8, 0x2F000060
.long 0xD3C40010, 0x04423959
.long 0xD86C0040, 0x30000060
.long 0xD86C10C0, 0x31000060
.long 0xD86C0048, 0x32000060
.long 0xD86C10C8, 0x33000060
.long 0xD3C40000, 0x04023B5A
.long 0xD86C0050, 0x34000060
.long 0xD86C10D0, 0x35000060
.long 0xD86C0058, 0x36000060
.long 0xD86C10D8, 0x37000060
.long 0xD3C40010, 0x04423B5B
.long 0xD86C0060, 0x38000060
.long 0xD86C10E0, 0x39000060
.long 0xD86C0068, 0x3A000060
.long 0xD86C10E8, 0x3B000060
.long 0xD3C40000, 0x04023D5C
.long 0xD86C0070, 0x3C000060
.long 0xD86C10F0, 0x3D000060
.long 0xD86C0078, 0x3E000060
.long 0xD86C10F8, 0x3F000060
.long 0xD3C40010, 0x04423D5D
.long 0xBF8A0000
.long 0xD86C0000, 0x00000062
.long 0xD86C0008, 0x01000062
.long 0xD86C0010, 0x02000062
.long 0xD86C0018, 0x03000062
.long 0xD3C40000, 0x04023F5E
.long 0xD86C0020, 0x04000062
.long 0xD86C0028, 0x05000062
.long 0xD86C0030, 0x06000062
.long 0xD86C0038, 0x07000062
.long 0xD3C40010, 0x04423F5F
.long 0xD86C0040, 0x08000062
.long 0xD86C0048, 0x09000062
.long 0xD86C0050, 0x0A000062
.long 0xD86C0058, 0x0B000062
.long 0xBF8F0001
.long 0x802E812E
.long 0xBF8CCA7F
.long 0xD3C40000, 0x04020120
.long 0xD86C0060, 0x0C000062
.long 0xD86C0068, 0x0D000062
.long 0xD3C40010, 0x04420121
.long 0xD86C0070, 0x0E000062
.long 0xD86C0078, 0x0F000062
.long 0xD3C40000, 0x04020322
.long 0xD3C40010, 0x04420323
.long 0xBF8A0000
.long 0xBF8CC87F
.long 0xD3C40000, 0x04020524
.long 0xD3C40010, 0x04420525
.long 0xD3C40000, 0x04020726
.long 0xD3C40010, 0x04420727
.long 0xD3C40000, 0x04020928
.long 0xD3C40010, 0x04420929
.long 0xD3C40000, 0x04020B2A
.long 0xD3C40010, 0x04420B2B
.long 0xD3C40000, 0x04020D2C
.long 0xD3C40010, 0x04420D2D
.long 0xD3C40000, 0x04020F2E
.long 0xD3C40010, 0x04420F2F
.long 0xBF8CC07F
.long 0xD3C40000, 0x04021130
.long 0xD3C40010, 0x04421131
.long 0xD3C40000, 0x04021332
.long 0xD3C40010, 0x04421333
.long 0xBF8F0000
.long 0xD3C40000, 0x04021534
.long 0xD3C40010, 0x04421535
.long 0xBF8A0000
.long 0xD86C0000, 0x40000061
.long 0xD86C1080, 0x41000061
.long 0xD86C0008, 0x42000061
.long 0xD86C1088, 0x43000061
.long 0xD3C40000, 0x04021736
.long 0xD86C0010, 0x44000061
.long 0xD86C1090, 0x45000061
.long 0xD86C0018, 0x46000061
.long 0xD86C1098, 0x47000061
.long 0xD3C40010, 0x04421737
.long 0xD86C0020, 0x48000061
.long 0xD86C10A0, 0x49000061
.long 0xD86C0028, 0x4A000061
.long 0xD86C10A8, 0x4B000061
.long 0xD3C40000, 0x04021938
.long 0xD86C0030, 0x4C000061
.long 0xD86C10B0, 0x4D000061
.long 0xD86C0038, 0x4E000061
.long 0xD86C10B8, 0x4F000061
.long 0xD3C40010, 0x04421939
.long 0xD86C0040, 0x50000061
.long 0xD86C10C0, 0x51000061
.long 0xD86C0048, 0x52000061
.long 0xD86C10C8, 0x53000061
.long 0xD3C40000, 0x04021B3A
.long 0xD86C0050, 0x54000061
.long 0xD86C10D0, 0x55000061
.long 0xD86C0058, 0x56000061
.long 0xD86C10D8, 0x57000061
.long 0xD3C40010, 0x04421B3B
.long 0xD86C0060, 0x58000061
.long 0xD86C10E0, 0x59000061
.long 0xD86C0068, 0x5A000061
.long 0xD86C10E8, 0x5B000061
.long 0xD3C40000, 0x04021D3C
.long 0xD86C0070, 0x5C000061
.long 0xD86C10F0, 0x5D000061
.long 0xD86C0078, 0x5E000061
.long 0xD86C10F8, 0x5F000061
.long 0xD3C40010, 0x04421D3D
.long 0xBF8A0000
.long 0xD86C0000, 0x10000063
.long 0xD86C0008, 0x11000063
.long 0xD3C40000, 0x04021F3E
.long 0xD86C0010, 0x12000063
.long 0xD86C0018, 0x13000063
.long 0xD86C0020, 0x14000063
.long 0xD86C0028, 0x15000063
.long 0xD86C0030, 0x16000063
.long 0xD3C40010, 0x04421F3F
.long 0xD86C0038, 0x17000063
.long 0xD86C0040, 0x18000063
.long 0xD86C0048, 0x19000063
.long 0xD86C0050, 0x1A000063
.long 0xD86C0058, 0x1B000063
.long 0xBF8F0001
.long 0x802E812E
.long 0xBF8CCA7F
.long 0xD3C40000, 0x04022140
.long 0xD86C0060, 0x1C000063
.long 0xD86C0068, 0x1D000063
.long 0xD3C40010, 0x04422141
.long 0xD86C0070, 0x1E000063
.long 0xD86C0078, 0x1F000063
.long 0xD3C40000, 0x04022342
.long 0xD3C40010, 0x04422343
.long 0xBF8A0000
.long 0xBF8CC87F
.long 0xD3C40000, 0x04022544
.long 0xD3C40010, 0x04422545
.long 0xD3C40000, 0x04022746
.long 0xD3C40010, 0x04422747
.long 0xD3C40000, 0x04022948
.long 0xD3C40010, 0x04422949
.long 0xD3C40000, 0x04022B4A
.long 0xD3C40010, 0x04422B4B
.long 0xD3C40000, 0x04022D4C
.long 0xD3C40010, 0x04422D4D
.long 0xD3C40000, 0x04022F4E
.long 0xD3C40010, 0x04422F4F
.long 0xBF8CC07F
.long 0xD3C40000, 0x04023150
.long 0xD3C40010, 0x04423151
.long 0xD3C40000, 0x04023352
.long 0xD3C40010, 0x04423353
.long 0xBF8F0000
.long 0xD3C40000, 0x04023554
.long 0xD3C40010, 0x04423555
.long 0xBF8A0000
.long 0xD86C0000, 0x20000060
.long 0xD86C1080, 0x21000060
.long 0xD86C0008, 0x22000060
.long 0xD86C1088, 0x23000060
.long 0xD3C40000, 0x04023756
.long 0xD86C0010, 0x24000060
.long 0xD86C1090, 0x25000060
.long 0xD86C0018, 0x26000060
.long 0xD86C1098, 0x27000060
.long 0xD3C40010, 0x04423757
.long 0xD86C0020, 0x28000060
.long 0xD86C10A0, 0x29000060
.long 0xD86C0028, 0x2A000060
.long 0xD86C10A8, 0x2B000060
.long 0xD3C40000, 0x04023958
.long 0xD86C0030, 0x2C000060
.long 0xD86C10B0, 0x2D000060
.long 0xD86C0038, 0x2E000060
.long 0xD86C10B8, 0x2F000060
.long 0xD3C40010, 0x04423959
.long 0xD86C0040, 0x30000060
.long 0xD86C10C0, 0x31000060
.long 0xD86C0048, 0x32000060
.long 0xD86C10C8, 0x33000060
.long 0xD3C40000, 0x04023B5A
.long 0xD86C0050, 0x34000060
.long 0xD86C10D0, 0x35000060
.long 0xD86C0058, 0x36000060
.long 0xD86C10D8, 0x37000060
.long 0xD3C40010, 0x04423B5B
.long 0xD86C0060, 0x38000060
.long 0xD86C10E0, 0x39000060
.long 0xD86C0068, 0x3A000060
.long 0xD86C10E8, 0x3B000060
.long 0xD3C40000, 0x04023D5C
.long 0xD86C0070, 0x3C000060
.long 0xD86C10F0, 0x3D000060
.long 0xD86C0078, 0x3E000060
.long 0xD86C10F8, 0x3F000060
.long 0xD3C40010, 0x04423D5D
.long 0xBF8A0000
.long 0xD86C0000, 0x00000062
.long 0xD86C0008, 0x01000062
.long 0xD3C40000, 0x04023F5E
.long 0xD86C0010, 0x02000062
.long 0xD86C0018, 0x03000062
.long 0xD86C0020, 0x04000062
.long 0xD86C0028, 0x05000062
.long 0xD86C0030, 0x06000062
.long 0xD3C40010, 0x04423F5F
.long 0xD86C0038, 0x07000062
.long 0xD86C0040, 0x08000062
.long 0xD86C0048, 0x09000062
.long 0xD86C0050, 0x0A000062
.long 0xD86C0058, 0x0B000062
.long 0xBF8F0001
.long 0x802E812E
.long 0xBF00C22E
.long 0xBF84FEAC
.long 0xBF8CCA7F
.long 0xD3C40000, 0x04020120
.long 0xD3C40010, 0x04420121
.long 0xD86C0060, 0x0C000062
.long 0xD86C0068, 0x0D000062
.long 0xD86C0070, 0x0E000062
.long 0xD86C0078, 0x0F000062
.long 0xD3C40000, 0x04020322
.long 0xD3C40010, 0x04420323
.long 0xBF8CCC7F
.long 0xD3C40000, 0x04020524
.long 0xD3C40010, 0x04420525
.long 0xD3C40000, 0x04020726
.long 0xD3C40010, 0x04420727
.long 0xBF8CCA7F
.long 0xD3C40000, 0x04020928
.long 0xD3C40010, 0x04420929
.long 0xD3C40000, 0x04020B2A
.long 0xD3C40010, 0x04420B2B
.long 0xBF8CC87F
.long 0xD3C40000, 0x04020D2C
.long 0xD3C40010, 0x04420D2D
.long 0xD3C40000, 0x04020F2E
.long 0xD3C40010, 0x04420F2F
.long 0xBF8CC07F
.long 0xD3C40000, 0x04021130
.long 0xBF068029
.long 0xBF850008
.long 0xE05C1000, 0x80052068
.long 0xE05C1020, 0x80052468
.long 0xE05C1040, 0x80052868
.long 0xE05C1060, 0x80052C68
.long 0xD3C40010, 0x04421131
.long 0xD3C40000, 0x04021332
.long 0xD3C40010, 0x04421333
.long 0xD3C40000, 0x04021534
.long 0xD3C40010, 0x04421535
.long 0xBF8A0000
.long 0xD86C0000, 0x40000061
.long 0xD86C1080, 0x41000061
.long 0xD86C0008, 0x42000061
.long 0xD86C1088, 0x43000061
.long 0xD3C40000, 0x04021736
.long 0xD86C0010, 0x44000061
.long 0xD86C1090, 0x45000061
.long 0xD86C0018, 0x46000061
.long 0xD86C1098, 0x47000061
.long 0xD3C40010, 0x04421737
.long 0xD86C0020, 0x48000061
.long 0xD86C10A0, 0x49000061
.long 0xD86C0028, 0x4A000061
.long 0xD86C10A8, 0x4B000061
.long 0xD3C40000, 0x04021938
.long 0xD86C0030, 0x4C000061
.long 0xD86C10B0, 0x4D000061
.long 0xD86C0038, 0x4E000061
.long 0xD86C10B8, 0x4F000061
.long 0xD3C40010, 0x04421939
.long 0xD86C0040, 0x50000061
.long 0xD86C10C0, 0x51000061
.long 0xD86C0048, 0x52000061
.long 0xD86C10C8, 0x53000061
.long 0xD3C40000, 0x04021B3A
.long 0xD86C0050, 0x54000061
.long 0xD86C10D0, 0x55000061
.long 0xD86C0058, 0x56000061
.long 0xD86C10D8, 0x57000061
.long 0xD3C40010, 0x04421B3B
.long 0xD86C0060, 0x58000061
.long 0xD86C10E0, 0x59000061
.long 0xD86C0068, 0x5A000061
.long 0xD86C10E8, 0x5B000061
.long 0xD3C40000, 0x04021D3C
.long 0xD86C0070, 0x5C000061
.long 0xD86C10F0, 0x5D000061
.long 0xD86C0078, 0x5E000061
.long 0xD86C10F8, 0x5F000061
.long 0xD3C40010, 0x04421D3D
.long 0xBF8A0000
.long 0xD86C0000, 0x10000063
.long 0xD86C0008, 0x11000063
.long 0xD3C40000, 0x04021F3E
.long 0xD86C0010, 0x12000063
.long 0xD86C0018, 0x13000063
.long 0xD86C0020, 0x14000063
.long 0xD86C0028, 0x15000063
.long 0xD86C0030, 0x16000063
.long 0xD3C40010, 0x04421F3F
.long 0xD86C0038, 0x17000063
.long 0xD86C0040, 0x18000063
.long 0xD86C0048, 0x19000063
.long 0xD86C0050, 0x1A000063
.long 0xD86C0058, 0x1B000063
.long 0xBF8CCA7F
.long 0xD3C40000, 0x04022140
.long 0xBF068029
.long 0xBF850008
.long 0xE05C1080, 0x80053068
.long 0xE05C10A0, 0x80053468
.long 0xE05C10C0, 0x80053868
.long 0xE05C10E0, 0x80053C68
.long 0xD3C40010, 0x04422141
.long 0xD86C0060, 0x1C000063
.long 0xD86C0068, 0x1D000063
.long 0xD86C0070, 0x1E000063
.long 0xD86C0078, 0x1F000063
.long 0xD3C40000, 0x04022342
.long 0xD3C40010, 0x04422343
.long 0xBF8CCC7F
.long 0xD3C40000, 0x04022544
.long 0xD3C40010, 0x04422545
.long 0xD3C40000, 0x04022746
.long 0xD3C40010, 0x04422747
.long 0xBF8CCA7F
.long 0xD3C40000, 0x04022948
.long 0xD3C40010, 0x04422949
.long 0xD3C40000, 0x04022B4A
.long 0xD3C40010, 0x04422B4B
.long 0xBF8CC87F
.long 0xD3C40000, 0x04022D4C
.long 0xD3C40010, 0x04422D4D
.long 0xD3C40000, 0x04022F4E
.long 0xD3C40010, 0x04422F4F
.long 0xBF8CC07F
.long 0xD3C40000, 0x04023150
.long 0xD3C40000, 0x04023352
.long 0xD3C40000, 0x04023554
.long 0xD3C40000, 0x04023756
.long 0xD3C40000, 0x04023958
.long 0xD3C40000, 0x04023B5A
.long 0xD3C40000, 0x04023D5C
.long 0xD3C40000, 0x04023F5E
.long 0xD3C40010, 0x04423151
.long 0xD3C40010, 0x04423353
.long 0xD3C40010, 0x04423555
.long 0xD3C40010, 0x04423757
.long 0xD3C40010, 0x04423959
.long 0xD3C40010, 0x04423B5B
.long 0xD3C40010, 0x04423D5D
.long 0xD3C40010, 0x04423F5F
.long 0xD3D84000, 0x18000100
.long 0xD3D84001, 0x18000101
.long 0xD3D84002, 0x18000102
.long 0xD3D84003, 0x18000103
.long 0xD3D84004, 0x18000104
.long 0xD3D84005, 0x18000105
.long 0xD3D84006, 0x18000106
.long 0xD3D84007, 0x18000107
.long 0xD3D84008, 0x18000108
.long 0xD3D84009, 0x18000109
.long 0xD3D8400A, 0x1800010A
.long 0xD3D8400B, 0x1800010B
.long 0xD3D8400C, 0x1800010C
.long 0xD3D8400D, 0x1800010D
.long 0xD3D8400E, 0x1800010E
.long 0xD3D8400F, 0x1800010F
.long 0xBF068029
.long 0xBF840032
.long 0xE07C1000, 0x80050068
.long 0xE07C1020, 0x80050468
.long 0xE07C1040, 0x80050868
.long 0xE07C1060, 0x80050C68
.long 0xD3D84000, 0x18000110
.long 0xD3D84001, 0x18000111
.long 0xD3D84002, 0x18000112
.long 0xD3D84003, 0x18000113
.long 0xD3D84004, 0x18000114
.long 0xD3D84005, 0x18000115
.long 0xD3D84006, 0x18000116
.long 0xD3D84007, 0x18000117
.long 0xD3D84008, 0x18000118
.long 0xD3D84009, 0x18000119
.long 0xD3D8400A, 0x1800011A
.long 0xD3D8400B, 0x1800011B
.long 0xD3D8400C, 0x1800011C
.long 0xD3D8400D, 0x1800011D
.long 0xD3D8400E, 0x1800011E
.long 0xD3D8400F, 0x1800011F
.long 0xE07C1080, 0x80050068
.long 0xE07C10A0, 0x80050468
.long 0xE07C10C0, 0x80050868
.long 0xE07C10E0, 0x80050C68
.long 0xBF8C0000
.long 0xBF810000
.long 0xBF8C0F74
.long 0xD1160000, 0x00005320
.long 0xD1160001, 0x00005321
.long 0xD1160002, 0x00005322
.long 0xD1160003, 0x00005323
.long 0xE07C1000, 0x80050068
.long 0xD1160004, 0x00005324
.long 0xD1160005, 0x00005325
.long 0xD1160006, 0x00005326
.long 0xD1160007, 0x00005327
.long 0xE07C1020, 0x80050468
.long 0xD1160008, 0x00005328
.long 0xD1160009, 0x00005329
.long 0xD116000A, 0x0000532A
.long 0xD116000B, 0x0000532B
.long 0xE07C1040, 0x80050868
.long 0xD116000C, 0x0000532C
.long 0xD116000D, 0x0000532D
.long 0xD116000E, 0x0000532E
.long 0xD116000F, 0x0000532F
.long 0xE07C1060, 0x80050C68
.long 0xD3D84000, 0x18000110
.long 0xD3D84001, 0x18000111
.long 0xD3D84002, 0x18000112
.long 0xD3D84003, 0x18000113
.long 0xD3D84004, 0x18000114
.long 0xD3D84005, 0x18000115
.long 0xD3D84006, 0x18000116
.long 0xD3D84007, 0x18000117
.long 0xD3D84008, 0x18000118
.long 0xD3D84009, 0x18000119
.long 0xD3D8400A, 0x1800011A
.long 0xD3D8400B, 0x1800011B
.long 0xD3D8400C, 0x1800011C
.long 0xD3D8400D, 0x1800011D
.long 0xD3D8400E, 0x1800011E
.long 0xD3D8400F, 0x1800011F
.long 0xBF8C0F70
.long 0xD1160000, 0x00005330
.long 0xD1160001, 0x00005331
.long 0xD1160002, 0x00005332
.long 0xD1160003, 0x00005333
.long 0xE07C1080, 0x80050068
.long 0xD1160004, 0x00005334
.long 0xD1160005, 0x00005335
.long 0xD1160006, 0x00005336
.long 0xD1160007, 0x00005337
.long 0xE07C10A0, 0x80050468
.long 0xD1160008, 0x00005338
.long 0xD1160009, 0x00005339
.long 0xD116000A, 0x0000533A
.long 0xD116000B, 0x0000533B
.long 0xE07C10C0, 0x80050868
.long 0xD116000C, 0x0000533C
.long 0xD116000D, 0x0000533D
.long 0xD116000E, 0x0000533E
.long 0xD116000F, 0x0000533F
.long 0xE07C10E0, 0x80050C68
.long 0xBF8C0000
.long 0xBF810000

