; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, ptr %8, 1
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 %9, 2
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 %10, 3, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %12, 4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %11, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %13, 4, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %1, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %2, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %3, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %5, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %4, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %6, 4, 1
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
  br label %42

42:                                               ; preds = %55, %14
  %43 = phi i64 [ 0, %14 ], [ %56, %55 ]
  %44 = icmp slt i64 %43, 2048
  br i1 %44, label %45, label %57

45:                                               ; preds = %42
  br label %46

46:                                               ; preds = %49, %45
  %47 = phi i64 [ 0, %45 ], [ %54, %49 ]
  %48 = icmp slt i64 %47, 2048
  br i1 %48, label %49, label %55

49:                                               ; preds = %46
  %50 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  %51 = mul nuw nsw i64 %43, 2048
  %52 = add nuw nsw i64 %51, %47
  %53 = getelementptr inbounds nuw float, ptr %50, i64 %52
  store float 0.000000e+00, ptr %53, align 4
  %54 = add i64 %47, 1
  br label %46

55:                                               ; preds = %46
  %56 = add i64 %43, 1
  br label %42

57:                                               ; preds = %42
  br label %58

58:                                               ; preds = %183, %57
  %59 = phi i64 [ 0, %57 ], [ %184, %183 ]
  %60 = icmp slt i64 %59, 2048
  br i1 %60, label %61, label %185

61:                                               ; preds = %58
  br label %62

62:                                               ; preds = %181, %61
  %63 = phi i64 [ 0, %61 ], [ %182, %181 ]
  %64 = icmp slt i64 %63, 2048
  br i1 %64, label %65, label %183

65:                                               ; preds = %62
  br label %66

66:                                               ; preds = %179, %65
  %67 = phi i64 [ 0, %65 ], [ %180, %179 ]
  %68 = icmp slt i64 %67, 2048
  br i1 %68, label %69, label %181

69:                                               ; preds = %66
  %70 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 0
  %71 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %72 = insertvalue { ptr, ptr, i64 } poison, ptr %70, 0
  %73 = insertvalue { ptr, ptr, i64 } %72, ptr %71, 1
  %74 = insertvalue { ptr, ptr, i64 } %73, i64 0, 2
  %75 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 2
  %76 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 3, 0
  %77 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 3, 1
  %78 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 1
  %80 = mul nsw i64 %59, %78
  %81 = add i64 %75, %80
  %82 = mul nsw i64 %63, %79
  %83 = add i64 %81, %82
  %84 = extractvalue { ptr, ptr, i64 } %74, 0
  %85 = extractvalue { ptr, ptr, i64 } %74, 1
  %86 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %84, 0
  %87 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %86, ptr %85, 1
  %88 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, i64 %83, 2
  %89 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, i64 64, 3, 0
  %90 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, i64 %78, 4, 0
  %91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, i64 64, 3, 1
  %92 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %91, i64 %79, 4, 1
  %93 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 0
  %94 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %95 = insertvalue { ptr, ptr, i64 } poison, ptr %93, 0
  %96 = insertvalue { ptr, ptr, i64 } %95, ptr %94, 1
  %97 = insertvalue { ptr, ptr, i64 } %96, i64 0, 2
  %98 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 2
  %99 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 0
  %100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 1
  %101 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 0
  %102 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 1
  %103 = mul nsw i64 %63, %101
  %104 = add i64 %98, %103
  %105 = mul nsw i64 %67, %102
  %106 = add i64 %104, %105
  %107 = extractvalue { ptr, ptr, i64 } %97, 0
  %108 = extractvalue { ptr, ptr, i64 } %97, 1
  %109 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %107, 0
  %110 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %109, ptr %108, 1
  %111 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %110, i64 %106, 2
  %112 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %111, i64 64, 3, 0
  %113 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %112, i64 %101, 4, 0
  %114 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %113, i64 64, 3, 1
  %115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, i64 %102, 4, 1
  %116 = mul nsw i64 %59, 2048
  %117 = add i64 %116, %67
  %118 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 0
  %119 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, 1
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %118, 0
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, ptr %119, 1
  %122 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %121, i64 %117, 2
  %123 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %122, i64 64, 3, 0
  %124 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %123, i64 2048, 4, 0
  %125 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %124, i64 64, 3, 1
  %126 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, i64 1, 4, 1
  br label %127

127:                                              ; preds = %177, %69
  %128 = phi i64 [ 0, %69 ], [ %178, %177 ]
  %129 = icmp slt i64 %128, 64
  br i1 %129, label %130, label %179

130:                                              ; preds = %127
  br label %131

131:                                              ; preds = %175, %130
  %132 = phi i64 [ 0, %130 ], [ %176, %175 ]
  %133 = icmp slt i64 %132, 64
  br i1 %133, label %134, label %177

134:                                              ; preds = %131
  br label %135

135:                                              ; preds = %138, %134
  %136 = phi i64 [ 0, %134 ], [ %174, %138 ]
  %137 = icmp slt i64 %136, 64
  br i1 %137, label %138, label %175

138:                                              ; preds = %135
  %139 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, 1
  %140 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, 2
  %141 = getelementptr float, ptr %139, i64 %140
  %142 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, 4, 0
  %143 = mul nuw nsw i64 %128, %142
  %144 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, 4, 1
  %145 = mul nuw nsw i64 %132, %144
  %146 = add nuw nsw i64 %143, %145
  %147 = getelementptr inbounds nuw float, ptr %141, i64 %146
  %148 = load float, ptr %147, align 4
  %149 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, 1
  %150 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, 2
  %151 = getelementptr float, ptr %149, i64 %150
  %152 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, 4, 0
  %153 = mul nuw nsw i64 %132, %152
  %154 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, 4, 1
  %155 = mul nuw nsw i64 %136, %154
  %156 = add nuw nsw i64 %153, %155
  %157 = getelementptr inbounds nuw float, ptr %151, i64 %156
  %158 = load float, ptr %157, align 4
  %159 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, 1
  %160 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, 2
  %161 = getelementptr float, ptr %159, i64 %160
  %162 = mul nuw nsw i64 %128, 2048
  %163 = add nuw nsw i64 %162, %136
  %164 = getelementptr inbounds nuw float, ptr %161, i64 %163
  %165 = load float, ptr %164, align 4
  %166 = fmul float %148, %158
  %167 = fadd float %165, %166
  %168 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, 1
  %169 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, 2
  %170 = getelementptr float, ptr %168, i64 %169
  %171 = mul nuw nsw i64 %128, 2048
  %172 = add nuw nsw i64 %171, %136
  %173 = getelementptr inbounds nuw float, ptr %170, i64 %172
  store float %167, ptr %173, align 4
  %174 = add i64 %136, 1
  br label %135

175:                                              ; preds = %135
  %176 = add i64 %132, 1
  br label %131

177:                                              ; preds = %131
  %178 = add i64 %128, 1
  br label %127

179:                                              ; preds = %127
  %180 = add i64 %67, 64
  br label %66

181:                                              ; preds = %66
  %182 = add i64 %63, 64
  br label %62

183:                                              ; preds = %62
  %184 = add i64 %59, 64
  br label %58

185:                                              ; preds = %58
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %41
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

