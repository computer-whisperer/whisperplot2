use std::collections::HashMap;

/// Numeric type descriptor for data storage and shader interpretation.
/// MVP: keep it minimal but forward‑looking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    /// Wall time in microseconds (semantic). Stored as u64 on CPU.
    U64TimeMicros,
}

impl DType {
    /// Map to a compact integer id for WGSL side (future use).
    pub fn wgsl_id(self) -> u32 {
        match self {
            DType::F32 => 1,
            DType::U64TimeMicros => 10,
        }
    }
}

/// A single immutable series of samples.
/// NOTE: MVP structure. Will be replaced by a pluggable DataSource interface
/// that can represent queried/mmap/streamed sources of diverse dtypes.
pub struct Series {
    pub name: String,
    pub x_dtype: DType,
    pub y_dtype: DType,
    /// Optional explicit x values (e.g., timestamps). If None, x is implicit index.
    pub x_raw_u64: Option<Vec<u64>>, // MVP keeps raw on CPU; GPU gets normalized f32 positions
    pub y_f32: Vec<f32>,
    // Cached immutable bounds computed at load time.
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f32,
    pub max_y: f32,
}

impl Series {
    pub fn from_implicit_y(name: impl Into<String>, y: Vec<f32>) -> Self {
        let name = name.into();
        let len = y.len();
        // For implicit x, min_x = 0, max_x = len-1
        let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in &y {
            if *v < min_y { min_y = *v; }
            if *v > max_y { max_y = *v; }
        }
        Series {
            name,
            x_dtype: DType::U64TimeMicros, // semantic placeholder; implicit index has integer semantics
            y_dtype: DType::F32,
            x_raw_u64: None,
            y_f32: y,
            min_x: 0.0,
            max_x: if len == 0 { 0.0 } else { (len - 1) as f64 },
            min_y,
            max_y,
        }
    }
}

/// A mutable owner/registry for loaded series. Series themselves are immutable
/// once inserted; Dataset manages adding/removing and cached global bounds.
pub struct Dataset {
    series: HashMap<String, Series>,
    pub full_min_x: f64,
    pub full_max_x: f64,
    pub full_min_y: f32,
    pub full_max_y: f32,
}

impl Dataset {
    pub fn new() -> Self {
        Dataset {
            series: HashMap::new(),
            full_min_x: 0.0,
            full_max_x: 0.0,
            full_min_y: 0.0,
            full_max_y: 0.0,
        }
    }

    pub fn insert(&mut self, key: impl Into<String>, series: Series) {
        self.series.insert(key.into(), series);
        self.recompute_bounds();
    }

    #[allow(dead_code)]
    pub fn remove(&mut self, key: &str) -> Option<Series> {
        let out = self.series.remove(key);
        if out.is_some() { self.recompute_bounds(); }
        out
    }

    pub fn get(&self, key: &str) -> Option<&Series> { self.series.get(key) }

    pub fn first_series(&self) -> Option<&Series> {
        // MVP convenience: any series (stable order not guaranteed)
        self.series.values().next()
    }

    pub fn len(&self) -> usize { self.series.len() }

    fn recompute_bounds(&mut self) {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for s in self.series.values() {
            if s.min_x < min_x { min_x = s.min_x; }
            if s.max_x > max_x { max_x = s.max_x; }
            if s.min_y < min_y { min_y = s.min_y; }
            if s.max_y > max_y { max_y = s.max_y; }
        }
        if self.series.is_empty() {
            min_x = 0.0; max_x = 0.0; min_y = 0.0; max_y = 0.0;
        }
        self.full_min_x = min_x;
        self.full_max_x = max_x;
        self.full_min_y = min_y;
        self.full_max_y = max_y;
    }
}
