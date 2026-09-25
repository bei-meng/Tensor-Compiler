; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) {
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %14, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %15, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %16, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %17, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %19, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %18, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %20, 4, 1
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %8, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %9, 2
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %10, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %12, 4, 0
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %11, 3, 1
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, i64 %13, 4, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, ptr %1, 1
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 %2, 2
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 %3, 3, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %5, 4, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %4, 3, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %6, 4, 1
  %43 = call ptr @malloc(i64 16777280)
  %44 = ptrtoint ptr %43 to i64
  %45 = add i64 %44, 63
  %46 = urem i64 %45, 64
  %47 = sub i64 %45, %46
  %48 = inttoptr i64 %47 to ptr
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %43, 0
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, ptr %48, 1
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 0, 2
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 2048, 3, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 2048, 3, 1
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 2048, 4, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 1, 4, 1
  %56 = call ptr @malloc(i64 16777280)
  %57 = ptrtoint ptr %56 to i64
  %58 = add i64 %57, 63
  %59 = urem i64 %58, 64
  %60 = sub i64 %58, %59
  %61 = inttoptr i64 %60 to ptr
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %56, 0
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, ptr %61, 1
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 0, 2
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 2048, 3, 0
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 2048, 3, 1
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 2048, 4, 0
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, i64 1, 4, 1
  br label %69

69:                                               ; preds = %82, %21
  %70 = phi i64 [ 0, %21 ], [ %83, %82 ]
  %71 = icmp slt i64 %70, 2048
  br i1 %71, label %72, label %84

72:                                               ; preds = %69
  br label %73

73:                                               ; preds = %76, %72
  %74 = phi i64 [ 0, %72 ], [ %81, %76 ]
  %75 = icmp slt i64 %74, 2048
  br i1 %75, label %76, label %82

76:                                               ; preds = %73
  %77 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 1
  %78 = mul nuw nsw i64 %70, 2048
  %79 = add nuw nsw i64 %78, %74
  %80 = getelementptr inbounds nuw float, ptr %77, i64 %79
  store float 0.000000e+00, ptr %80, align 4
  %81 = add i64 %74, 1
  br label %73

82:                                               ; preds = %73
  %83 = add i64 %70, 1
  br label %69

84:                                               ; preds = %69
  br label %85

85:                                               ; preds = %210, %84
  %86 = phi i64 [ 0, %84 ], [ %211, %210 ]
  %87 = icmp slt i64 %86, 2048
  br i1 %87, label %88, label %212

88:                                               ; preds = %85
  br label %89

89:                                               ; preds = %208, %88
  %90 = phi i64 [ 0, %88 ], [ %209, %208 ]
  %91 = icmp slt i64 %90, 2048
  br i1 %91, label %92, label %210

92:                                               ; preds = %89
  br label %93

93:                                               ; preds = %206, %92
  %94 = phi i64 [ 0, %92 ], [ %207, %206 ]
  %95 = icmp slt i64 %94, 2048
  br i1 %95, label %96, label %208

96:                                               ; preds = %93
  %97 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 0
  %98 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 1
  %99 = insertvalue { ptr, ptr, i64 } poison, ptr %97, 0
  %100 = insertvalue { ptr, ptr, i64 } %99, ptr %98, 1
  %101 = insertvalue { ptr, ptr, i64 } %100, i64 0, 2
  %102 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 2
  %103 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 3, 0
  %104 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 3, 1
  %105 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 0
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, 4, 1
  %107 = mul nsw i64 %86, %105
  %108 = add i64 %102, %107
  %109 = mul nsw i64 %90, %106
  %110 = add i64 %108, %109
  %111 = extractvalue { ptr, ptr, i64 } %101, 0
  %112 = extractvalue { ptr, ptr, i64 } %101, 1
  %113 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %111, 0
  %114 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %113, ptr %112, 1
  %115 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %114, i64 %110, 2
  %116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %115, i64 64, 3, 0
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %116, i64 %105, 4, 0
  %118 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %117, i64 64, 3, 1
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %118, i64 %106, 4, 1
  %120 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 0
  %121 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 1
  %122 = insertvalue { ptr, ptr, i64 } poison, ptr %120, 0
  %123 = insertvalue { ptr, ptr, i64 } %122, ptr %121, 1
  %124 = insertvalue { ptr, ptr, i64 } %123, i64 0, 2
  %125 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 2
  %126 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 3, 0
  %127 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 3, 1
  %128 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 0
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, 4, 1
  %130 = mul nsw i64 %90, %128
  %131 = add i64 %125, %130
  %132 = mul nsw i64 %94, %129
  %133 = add i64 %131, %132
  %134 = extractvalue { ptr, ptr, i64 } %124, 0
  %135 = extractvalue { ptr, ptr, i64 } %124, 1
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %134, 0
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, ptr %135, 1
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 %133, 2
  %139 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, i64 64, 3, 0
  %140 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %139, i64 %128, 4, 0
  %141 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %140, i64 64, 3, 1
  %142 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %141, i64 %129, 4, 1
  %143 = mul nsw i64 %86, 2048
  %144 = add i64 %143, %94
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 0
  %146 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 1
  %147 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %145, 0
  %148 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %147, ptr %146, 1
  %149 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %148, i64 %144, 2
  %150 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %149, i64 64, 3, 0
  %151 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %150, i64 2048, 4, 0
  %152 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %151, i64 64, 3, 1
  %153 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %152, i64 1, 4, 1
  br label %154

154:                                              ; preds = %204, %96
  %155 = phi i64 [ 0, %96 ], [ %205, %204 ]
  %156 = icmp slt i64 %155, 64
  br i1 %156, label %157, label %206

157:                                              ; preds = %154
  br label %158

158:                                              ; preds = %202, %157
  %159 = phi i64 [ 0, %157 ], [ %203, %202 ]
  %160 = icmp slt i64 %159, 64
  br i1 %160, label %161, label %204

161:                                              ; preds = %158
  br label %162

162:                                              ; preds = %165, %161
  %163 = phi i64 [ 0, %161 ], [ %201, %165 ]
  %164 = icmp slt i64 %163, 64
  br i1 %164, label %165, label %202

165:                                              ; preds = %162
  %166 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 1
  %167 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 2
  %168 = getelementptr float, ptr %166, i64 %167
  %169 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 4, 0
  %170 = mul nuw nsw i64 %155, %169
  %171 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, 4, 1
  %172 = mul nuw nsw i64 %159, %171
  %173 = add nuw nsw i64 %170, %172
  %174 = getelementptr inbounds nuw float, ptr %168, i64 %173
  %175 = load float, ptr %174, align 4
  %176 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, 1
  %177 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, 2
  %178 = getelementptr float, ptr %176, i64 %177
  %179 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, 4, 0
  %180 = mul nuw nsw i64 %159, %179
  %181 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %142, 4, 1
  %182 = mul nuw nsw i64 %163, %181
  %183 = add nuw nsw i64 %180, %182
  %184 = getelementptr inbounds nuw float, ptr %178, i64 %183
  %185 = load float, ptr %184, align 4
  %186 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 1
  %187 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 2
  %188 = getelementptr float, ptr %186, i64 %187
  %189 = mul nuw nsw i64 %155, 2048
  %190 = add nuw nsw i64 %189, %163
  %191 = getelementptr inbounds nuw float, ptr %188, i64 %190
  %192 = load float, ptr %191, align 4
  %193 = fmul float %175, %185
  %194 = fadd float %192, %193
  %195 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 1
  %196 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %153, 2
  %197 = getelementptr float, ptr %195, i64 %196
  %198 = mul nuw nsw i64 %155, 2048
  %199 = add nuw nsw i64 %198, %163
  %200 = getelementptr inbounds nuw float, ptr %197, i64 %199
  store float %194, ptr %200, align 4
  %201 = add i64 %163, 1
  br label %162

202:                                              ; preds = %162
  %203 = add i64 %159, 1
  br label %158

204:                                              ; preds = %158
  %205 = add i64 %155, 1
  br label %154

206:                                              ; preds = %154
  %207 = add i64 %94, 64
  br label %93

208:                                              ; preds = %93
  %209 = add i64 %90, 64
  br label %89

210:                                              ; preds = %89
  %211 = add i64 %86, 64
  br label %85

212:                                              ; preds = %85
  br label %213

213:                                              ; preds = %242, %212
  %214 = phi i64 [ 0, %212 ], [ %243, %242 ]
  %215 = icmp slt i64 %214, 2048
  br i1 %215, label %216, label %244

216:                                              ; preds = %213
  br label %217

217:                                              ; preds = %220, %216
  %218 = phi i64 [ 0, %216 ], [ %241, %220 ]
  %219 = icmp slt i64 %218, 2048
  br i1 %219, label %220, label %242

220:                                              ; preds = %217
  %221 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, 1
  %222 = mul nuw nsw i64 %214, 2048
  %223 = add nuw nsw i64 %222, %218
  %224 = getelementptr inbounds nuw float, ptr %221, i64 %223
  %225 = load float, ptr %224, align 4
  %226 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %227 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 2
  %228 = getelementptr float, ptr %226, i64 %227
  %229 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %230 = mul nuw nsw i64 %214, %229
  %231 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 1
  %232 = mul nuw nsw i64 %218, %231
  %233 = add nuw nsw i64 %230, %232
  %234 = getelementptr inbounds nuw float, ptr %228, i64 %233
  %235 = load float, ptr %234, align 4
  %236 = fadd float %225, %235
  %237 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %238 = mul nuw nsw i64 %214, 2048
  %239 = add nuw nsw i64 %238, %218
  %240 = getelementptr inbounds nuw float, ptr %237, i64 %239
  store float %236, ptr %240, align 4
  %241 = add i64 %218, 1
  br label %217

242:                                              ; preds = %217
  %243 = add i64 %214, 1
  br label %213

244:                                              ; preds = %213
  br label %245

245:                                              ; preds = %264, %244
  %246 = phi i64 [ 0, %244 ], [ %265, %264 ]
  %247 = icmp slt i64 %246, 2048
  br i1 %247, label %248, label %266

248:                                              ; preds = %245
  br label %249

249:                                              ; preds = %252, %248
  %250 = phi i64 [ 0, %248 ], [ %263, %252 ]
  %251 = icmp slt i64 %250, 2048
  br i1 %251, label %252, label %264

252:                                              ; preds = %249
  %253 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %254 = mul nuw nsw i64 %246, 2048
  %255 = add nuw nsw i64 %254, %250
  %256 = getelementptr inbounds nuw float, ptr %253, i64 %255
  %257 = load float, ptr %256, align 4
  %258 = call float @llvm.maxnum.f32(float %257, float 0.000000e+00)
  %259 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, 1
  %260 = mul nuw nsw i64 %246, 2048
  %261 = add nuw nsw i64 %260, %250
  %262 = getelementptr inbounds nuw float, ptr %259, i64 %261
  store float %258, ptr %262, align 4
  %263 = add i64 %250, 1
  br label %249

264:                                              ; preds = %249
  %265 = add i64 %246, 1
  br label %245

266:                                              ; preds = %245
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %55
}

define void @_mlir_ciface_main(ptr %0, ptr %1, ptr %2, ptr %3) {
  %5 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 0
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 1
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 2
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 3, 0
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 3, 1
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, 4, 1
  %13 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 0
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 1
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 2
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 3, 0
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 3, 1
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 4, 0
  %20 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, 4, 1
  %21 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %3, align 8
  %22 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 0
  %23 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %24 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 2
  %25 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 0
  %26 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3, 1
  %27 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 0
  %28 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 1
  %29 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @main(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr %22, ptr %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, ptr %0, align 8
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #0

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

