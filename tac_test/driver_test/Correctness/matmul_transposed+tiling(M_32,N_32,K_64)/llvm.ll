; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, ptr %1, 1
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 %2, 2
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 %3, 3, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %5, 4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %4, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %6, 4, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %8, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %9, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %10, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %12, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %11, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %13, 4, 1
  %29 = call ptr @malloc(i64 16777280)
  %30 = ptrtoint ptr %29 to i64
  %31 = add i64 %30, 63
  %32 = urem i64 %31, 64
  %33 = sub i64 %31, %32
  %34 = inttoptr i64 %33 to ptr
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %29, 0
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, ptr %34, 1
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 0, 2
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 2048, 3, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 2048, 3, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 2048, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 1, 4, 1
  %42 = call ptr @malloc(i64 16777280)
  %43 = ptrtoint ptr %42 to i64
  %44 = add i64 %43, 63
  %45 = urem i64 %44, 64
  %46 = sub i64 %44, %45
  %47 = inttoptr i64 %46 to ptr
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %42, 0
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, ptr %47, 1
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, i64 0, 2
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 2048, 3, 0
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 2048, 3, 1
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 2048, 4, 0
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 1, 4, 1
  br label %55

55:                                               ; preds = %68, %14
  %56 = phi i64 [ 0, %14 ], [ %69, %68 ]
  %57 = icmp slt i64 %56, 2048
  br i1 %57, label %58, label %70

58:                                               ; preds = %55
  br label %59

59:                                               ; preds = %62, %58
  %60 = phi i64 [ 0, %58 ], [ %67, %62 ]
  %61 = icmp slt i64 %60, 2048
  br i1 %61, label %62, label %68

62:                                               ; preds = %59
  %63 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 1
  %64 = mul nuw nsw i64 %56, 2048
  %65 = add nuw nsw i64 %64, %60
  %66 = getelementptr inbounds nuw float, ptr %63, i64 %65
  store float 0.000000e+00, ptr %66, align 4
  %67 = add i64 %60, 1
  br label %59

68:                                               ; preds = %59
  %69 = add i64 %56, 1
  br label %55

70:                                               ; preds = %55
  br label %71

71:                                               ; preds = %94, %70
  %72 = phi i64 [ 0, %70 ], [ %95, %94 ]
  %73 = icmp slt i64 %72, 2048
  br i1 %73, label %74, label %96

74:                                               ; preds = %71
  br label %75

75:                                               ; preds = %78, %74
  %76 = phi i64 [ 0, %74 ], [ %93, %78 ]
  %77 = icmp slt i64 %76, 2048
  br i1 %77, label %78, label %94

78:                                               ; preds = %75
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %80 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 2
  %81 = getelementptr float, ptr %79, i64 %80
  %82 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %83 = mul nuw nsw i64 %76, %82
  %84 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 1
  %85 = mul nuw nsw i64 %72, %84
  %86 = add nuw nsw i64 %83, %85
  %87 = getelementptr inbounds nuw float, ptr %81, i64 %86
  %88 = load float, ptr %87, align 4
  %89 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  %90 = mul nuw nsw i64 %72, 2048
  %91 = add nuw nsw i64 %90, %76
  %92 = getelementptr inbounds nuw float, ptr %89, i64 %91
  store float %88, ptr %92, align 4
  %93 = add i64 %76, 1
  br label %75

94:                                               ; preds = %75
  %95 = add i64 %72, 1
  br label %71

96:                                               ; preds = %71
  br label %97

97:                                               ; preds = %207, %96
  %98 = phi i64 [ 0, %96 ], [ %208, %207 ]
  %99 = icmp slt i64 %98, 2048
  br i1 %99, label %100, label %209

100:                                              ; preds = %97
  br label %101

101:                                              ; preds = %205, %100
  %102 = phi i64 [ 0, %100 ], [ %206, %205 ]
  %103 = icmp slt i64 %102, 2048
  br i1 %103, label %104, label %207

104:                                              ; preds = %101
  br label %105

105:                                              ; preds = %203, %104
  %106 = phi i64 [ 0, %104 ], [ %204, %203 ]
  %107 = icmp slt i64 %106, 2048
  br i1 %107, label %108, label %205

108:                                              ; preds = %105
  %109 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 0
  %110 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %111 = insertvalue { ptr, ptr, i64 } poison, ptr %109, 0
  %112 = insertvalue { ptr, ptr, i64 } %111, ptr %110, 1
  %113 = insertvalue { ptr, ptr, i64 } %112, i64 0, 2
  %114 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 2
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 0
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 1
  %117 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 0
  %118 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 1
  %119 = mul nsw i64 %98, %117
  %120 = add i64 %114, %119
  %121 = mul nsw i64 %106, %118
  %122 = add i64 %120, %121
  %123 = extractvalue { ptr, ptr, i64 } %113, 0
  %124 = extractvalue { ptr, ptr, i64 } %113, 1
  %125 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %123, 0
  %126 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, ptr %124, 1
  %127 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, i64 %122, 2
  %128 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %127, i64 64, 3, 0
  %129 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %128, i64 %117, 4, 0
  %130 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %129, i64 64, 3, 1
  %131 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %130, i64 %118, 4, 1
  %132 = mul nsw i64 %102, 2048
  %133 = add i64 %132, %106
  %134 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 0
  %135 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %134, 0
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, ptr %135, 1
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 %133, 2
  %139 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, i64 64, 3, 0
  %140 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %139, i64 2048, 4, 0
  %141 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %140, i64 64, 3, 1
  %142 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %141, i64 1, 4, 1
  %143 = mul nsw i64 %98, 2048
  %144 = add i64 %143, %102
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 0
  %146 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, 1
  %147 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %145, 0
  %148 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %147, ptr %146, 1
  %149 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %148, i64 %144, 2
  %150 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %149, i64 64, 3, 0
  %151 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, i64 2048, 4, 0
  %152 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, i64 64, 3, 1
  %153 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, i64 1, 4, 1
  br label %154

154:                                              ; preds = %201, %108
  %155 = phi i64 [ 0, %108 ], [ %202, %201 ]
  %156 = icmp slt i64 %155, 64
  br i1 %156, label %157, label %203

157:                                              ; preds = %154
  br label %158

158:                                              ; preds = %199, %157
  %159 = phi i64 [ 0, %157 ], [ %200, %199 ]
  %160 = icmp slt i64 %159, 64
  br i1 %160, label %161, label %201

161:                                              ; preds = %158
  br label %162

162:                                              ; preds = %165, %161
  %163 = phi i64 [ 0, %161 ], [ %198, %165 ]
  %164 = icmp slt i64 %163, 64
  br i1 %164, label %165, label %199

165:                                              ; preds = %162
  %166 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, 1
  %167 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, 2
  %168 = getelementptr float, ptr %166, i64 %167
  %169 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, 4, 0
  %170 = mul nuw nsw i64 %155, %169
  %171 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %131, 4, 1
  %172 = mul nuw nsw i64 %163, %171
  %173 = add nuw nsw i64 %170, %172
  %174 = getelementptr inbounds nuw float, ptr %168, i64 %173
  %175 = load float, ptr %174, align 4
  %176 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, 1
  %177 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, 2
  %178 = getelementptr float, ptr %176, i64 %177
  %179 = mul nuw nsw i64 %159, 2048
  %180 = add nuw nsw i64 %179, %163
  %181 = getelementptr inbounds nuw float, ptr %178, i64 %180
  %182 = load float, ptr %181, align 4
  %183 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 1
  %184 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 2
  %185 = getelementptr float, ptr %183, i64 %184
  %186 = mul nuw nsw i64 %155, 2048
  %187 = add nuw nsw i64 %186, %159
  %188 = getelementptr inbounds nuw float, ptr %185, i64 %187
  %189 = load float, ptr %188, align 4
  %190 = fmul float %175, %182
  %191 = fadd float %189, %190
  %192 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 1
  %193 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 2
  %194 = getelementptr float, ptr %192, i64 %193
  %195 = mul nuw nsw i64 %155, 2048
  %196 = add nuw nsw i64 %195, %159
  %197 = getelementptr inbounds nuw float, ptr %194, i64 %196
  store float %191, ptr %197, align 4
  %198 = add i64 %163, 1
  br label %162

199:                                              ; preds = %162
  %200 = add i64 %159, 1
  br label %158

201:                                              ; preds = %158
  %202 = add i64 %155, 1
  br label %154

203:                                              ; preds = %154
  %204 = add i64 %106, 64
  br label %105

205:                                              ; preds = %105
  %206 = add i64 %102, 64
  br label %101

207:                                              ; preds = %101
  %208 = add i64 %98, 64
  br label %97

209:                                              ; preds = %97
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %54
}

define void @_mlir_ciface_main(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 3, 1
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 4, 1
  %12 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 0
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 1
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 2
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 0
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 1
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 0
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 1
  %20 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @main(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, ptr %0, align 8
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

