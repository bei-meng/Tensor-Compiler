# 定位三个缓存的路径（单核心 cpu0 视角，编译器优化用单核心容量）
L1I=$(grep -rl 'Instruction' /sys/devices/system/cpu/cpu0/cache/index*/type | sed 's|/type||')
L1D=$(grep -rl '^Data$' /sys/devices/system/cpu/cpu0/cache/index*/type | sed 's|/type||' | head -1)
LL=$(for d in /sys/devices/system/cpu/cpu0/cache/index*; do
  [ "$(cat "$d/type")" = "Unified" ] && echo "$(cat "$d/level") $d"
done | sort -rn | head -n1 | cut -d' ' -f2)

# 工具函数：字节数自动格式化为 KB/MB 可读单位
format_size() {
  local bytes=$1
  if [ "$bytes" -lt 1024 ]; then
    echo "${bytes}B"
  elif [ "$bytes" -lt $((1024 * 1024)) ]; then
    echo "$((bytes / 1024))KB"
  else
    echo "$((bytes / 1024 / 1024))MB"
  fi
}

# 工具函数：获取缓存完整参数（总字节数、相联路数、缓存行大小）
get_cache_details() {
  local path=$1
  local sets=$(cat "$path/number_of_sets")
  local ways=$(cat "$path/ways_of_associativity")
  local line=$(cat "$path/coherency_line_size")
  local total_bytes=$((sets * ways * line))
  echo "$total_bytes $ways $line"
}

# 读取并拆分各缓存参数
read i1_bytes i1_ways i1_line <<< $(get_cache_details "$L1I")
read d1_bytes d1_ways d1_line <<< $(get_cache_details "$L1D")
read ll_bytes ll_ways ll_line <<< $(get_cache_details "$LL")
ll_level=$(cat "$LL/level")

# 自动输出：参数值 + 注释全部动态生成
echo "--I1=$i1_bytes,$i1_ways,$i1_line   # Instruction L1:$(format_size $i1_bytes), $i1_ways-way, ${i1_line}B line"
echo "--D1=$d1_bytes,$d1_ways,$d1_line   # Data L1:$(format_size $d1_bytes), $d1_ways-way, ${d1_line}B line"
echo "--LL=$ll_bytes,$ll_ways,$ll_line   # Last Level (L$ll_level): $(format_size $ll_bytes), $ll_ways-way, ${ll_line}B line"
