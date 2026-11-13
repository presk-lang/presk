//! The `.psk.d` container: round trips, widths, malformed files, the pipeline.

use std::path::PathBuf;

use presk_data::{Layout, Leaf, Pipeline, Reader, Value, idx_width, lm_windows, shuffle_indices};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("presk-data-tests-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn tokens() -> Vec<u64> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/tests/data/tokens.txt");
    std::fs::read_to_string(root).unwrap().split_whitespace().map(|t| t.parse().unwrap()).collect()
}

#[test]
fn layouts_from_types() {
    let l = Layout::from_type_text("(idx[64][seq: 8], idx[64][seq: 8])").unwrap();
    assert_eq!(l, Layout::Tuple(vec![Layout::Leaf(Leaf::Idx { bound: 64, shape: vec![8] }), Layout::Leaf(Leaf::Idx { bound: 64, shape: vec![8] })]));
    assert_eq!(l.sample_bytes(), 16);
    let l = Layout::from_type_text("ℝ[batch: 2, model: 4]").unwrap();
    assert_eq!(l.sample_bytes(), 64);
    assert_eq!(Layout::from_type_text("Real[3]").unwrap().sample_bytes(), 24);
    assert_eq!(Layout::from_type_text("ℝ[]").unwrap().sample_bytes(), 8);
    for bad in ["ℝ[..., 4]", "ℝ[D]", "idx[0][2]", "params MiniGPT", "nat", "ℝ[4] ℝ[4]", "(("] {
        let e = Layout::from_type_text(bad).unwrap_err();
        assert_eq!(e.code, "E9001", "{bad}: {}", e.message);
    }
}

#[test]
fn idx_widths() {
    assert_eq!(idx_width(1), 1);
    assert_eq!(idx_width(256), 1);
    assert_eq!(idx_width(257), 2);
    assert_eq!(idx_width(65_536), 2);
    assert_eq!(idx_width(65_537), 4);
    assert_eq!(idx_width(1 << 40), 8);
}

#[test]
fn round_trip_windows() {
    let toks = tokens();
    let (ty, layout, samples) = lm_windows(&toks, 64, 8, 8).unwrap();
    assert_eq!(ty, "(idx[64][seq: 8], idx[64][seq: 8])");
    assert_eq!(samples.len(), (toks.len() - 8).div_ceil(8));
    // the target is the input shifted by one
    if let Value::Tuple(xy) = &samples[0] {
        assert_eq!(xy[0], Value::Idx(toks[0..8].to_vec()));
        assert_eq!(xy[1], Value::Idx(toks[1..9].to_vec()));
    } else {
        panic!()
    }
    let path = tmp("windows.psk.d");
    assert_eq!(presk_data::write(&path, &ty, &layout, &samples).unwrap(), samples.len() as u64);
    let mut r = Reader::open(&path).unwrap();
    assert_eq!(r.header().sample_ty, ty);
    assert_eq!(r.len(), samples.len());
    assert_eq!(r.read_all().unwrap(), samples);
    assert_eq!(r.get(3).unwrap(), samples[3]);
    assert_eq!(r.get(samples.len()).unwrap_err().code, "E9003");
    // overlapping windows
    let (_, _, dense) = lm_windows(&toks, 64, 8, 1).unwrap();
    assert_eq!(dense.len(), toks.len() - 8);
    assert_eq!(lm_windows(&toks, 32, 8, 8).unwrap_err().code, "E9004");
}

#[test]
fn round_trip_reals_and_text() {
    let layout = Layout::from_type_text("(ℝ[model: 2], idx[10][])").unwrap();
    let s1 = Value::parse_text(&layout, "0.5 -1.25 7").unwrap();
    assert_eq!(s1, Value::Tuple(vec![Value::Real(vec![0.5, -1.25]), Value::Idx(vec![7])]));
    assert_eq!(Value::parse_text(&layout, "0.5 -1.25 10").unwrap_err().code, "E9004");
    assert_eq!(Value::parse_text(&layout, "0.5 -1.25").unwrap_err().code, "E9004");
    assert_eq!(Value::parse_text(&layout, "0.5 -1.25 7 8").unwrap_err().code, "E9004");
    let path = tmp("reals.psk.d");
    presk_data::write(&path, "(ℝ[model: 2], idx[10][])", &layout, std::slice::from_ref(&s1)).unwrap();
    let mut r = Reader::open(&path).unwrap();
    assert_eq!(r.get(0).unwrap(), s1);
    assert_eq!(s1.render(&layout), "([0.5, -1.25], 7)");
    assert_eq!(s1.to_json(&layout).to_string(), "[[0.5,-1.25],7]");
}

#[test]
fn malformed_files() {
    let bad = tmp("bad_magic.psk.d");
    std::fs::write(&bad, b"NOPE\x01\x00").unwrap();
    assert_eq!(Reader::open(&bad).unwrap_err().code, "E9002");
    let short = tmp("short.psk.d");
    std::fs::write(&short, b"PS").unwrap();
    assert_eq!(Reader::open(&short).unwrap_err().code, "E9002");
    // a valid header announcing more samples than present
    let layout = Layout::from_type_text("idx[4][2]").unwrap();
    let ok = tmp("ok.psk.d");
    presk_data::write(&ok, "idx[4][2]", &layout, &[Value::Idx(vec![1, 2]), Value::Idx(vec![3, 0])]).unwrap();
    let mut bytes = std::fs::read(&ok).unwrap();
    bytes.truncate(bytes.len() - 1);
    let trunc = tmp("trunc.psk.d");
    std::fs::write(&trunc, &bytes).unwrap();
    assert_eq!(Reader::open(&trunc).unwrap_err().code, "E9003");
    // a value out of its bound is refused at write time
    assert_eq!(presk_data::write(&ok, "idx[4][2]", &layout, &[Value::Idx(vec![4, 0])]).unwrap_err().code, "E9004");
    assert_eq!(presk_data::write(&ok, "idx[4][2]", &layout, &[Value::Real(vec![0.0, 0.0])]).unwrap_err().code, "E9005");
}

#[test]
fn params_files_need_a_layout() {
    let layout = Layout::Record(vec![("w".into(), Layout::Leaf(Leaf::Real { shape: vec![2, 1] })), ("b".into(), Layout::Leaf(Leaf::Real { shape: vec![1] }))]);
    let theta = Value::Record(vec![("w".into(), Value::Real(vec![0.25, -0.5])), ("b".into(), Value::Real(vec![1.0]))]);
    let path = tmp("theta.psk.d");
    presk_data::write(&path, "params Linear{2, 1}", &layout, std::slice::from_ref(&theta)).unwrap();
    let r = Reader::open(&path).unwrap();
    assert_eq!(r.layout().unwrap_err().code, "E9001");
    let mut r = r.with_layout(layout.clone()).unwrap();
    assert_eq!(r.get(0).unwrap(), theta);
    assert_eq!(theta.render(&layout), "{w = [[0.25], [-0.5]], b = [1]}");
}

#[test]
fn pipeline() {
    let p = Pipeline { shuffle: Some(1234), batch: 4, take: None };
    let b1 = p.batches(10, 1);
    assert_eq!(b1.len(), 2, "incomplete last batch dropped");
    let mut seen: Vec<usize> = b1.iter().flatten().copied().collect();
    seen.sort();
    assert_eq!(seen.len(), 8);
    assert_eq!(p.batches(10, 1), b1, "deterministic");
    assert_ne!(p.batches(10, 2), b1, "re-keyed per epoch");
    let mut perm = shuffle_indices(50, 7, 3);
    perm.sort();
    assert_eq!(perm, (0..50).collect::<Vec<_>>());
    let q = Pipeline { shuffle: None, batch: 3, take: Some(2) };
    assert_eq!(q.batches(10, 1), vec![vec![0, 1, 2], vec![3, 4, 5]]);
    assert_eq!(q.batches_per_epoch(10), 2);
    assert_eq!(Pipeline { shuffle: None, batch: 32, take: None }.batches_per_epoch(100), 3);
}

#[test]
fn info_snapshot() {
    let toks = tokens();
    let (ty, layout, samples) = lm_windows(&toks, 64, 8, 8).unwrap();
    let path = tmp("info.psk.d");
    presk_data::write(&path, &ty, &layout, &samples).unwrap();
    let mut r = Reader::open(&path).unwrap();
    insta::assert_snapshot!(presk_data::info(&mut r, 2).unwrap());
}

/// The character tokenizer (0.4.0 step 2): the vocabulary is every distinct
/// character in sorted order, and encoding then decoding is the identity —
/// including on accents, apostrophes and newlines, which the Code civil has.
#[test]
fn character_vocabulary_round_trips() {
    let text = "Les lois et,\nlorsqu'ils sont publiés au Journal officiel de la République française…\r\n";
    let vocab = presk_data::char_vocab(text);
    assert!(vocab.iter().all(|t| t.chars().count() == 1), "one character per token");
    assert!(vocab.windows(2).all(|w| w[0] < w[1]), "sorted and distinct");
    assert!(vocab.iter().any(|t| t == "é") && vocab.iter().any(|t| t == "\n") && vocab.iter().any(|t| t == "…"));
    let tokens = presk_data::encode(text, &vocab).unwrap();
    assert_eq!(tokens.len(), text.chars().count(), "one token per character, not per byte");
    assert!(tokens.iter().all(|t| *t < vocab.len() as u64));
    assert_eq!(presk_data::decode_text(&tokens, &vocab).unwrap(), text);

    // A character the vocabulary does not hold, and a sub-word vocabulary.
    let e = presk_data::encode("Ω", &vocab).unwrap_err();
    assert_eq!(e.code, "E9004");
    assert!(e.message.contains("U+03A9"), "{}", e.message);
    let bpe = vec!["le".to_string(), "s".to_string()];
    assert_eq!(presk_data::encode("les", &bpe).unwrap_err().code, "E9004");
}

/// Version 2 (plan 0.17.0 step 7, D180): each real leaf in its encoding,
/// read back exactly; a file of `f64` leaves stays version 1, byte for byte
/// what `write` writes; a number that is not one of its leaf's encoding is
/// refused.
#[test]
fn version_two_encodings() {
    use presk_data::{Enc, write_encoded};
    let ty = "(ℝ[a: 3], ℝ[b: 2], idx[5][2], ℝ[])";
    let layout = Layout::from_type_text(ty).unwrap();
    // 1.5, −0.25 and 3 are numbers of bf16; 0.1f32 is one of f32 only.
    let v = Value::Tuple(vec![
        Value::Real(vec![1.5, -0.25, 3.0]),
        Value::Real(vec![f64::from(0.1f32), -2.0]),
        Value::Idx(vec![4, 0]),
        Value::Real(vec![0.1]),
    ]);
    let encs = [Enc::Bf16, Enc::F32, Enc::F64, Enc::F64];
    let path = tmp("v2.psk.d");
    write_encoded(&path, ty, &layout, &encs, std::slice::from_ref(&v)).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 2, "a narrower leaf: version 2");
    let mut r = Reader::open(&path).unwrap();
    assert_eq!(r.header().encodings, encs.to_vec());
    assert_eq!(r.get(0).unwrap(), v, "every value read back exactly");
    // header (4 + 2 + 4 + type + 8 + 4 + 4 leaves) + 3·2 + 2·4 + 2·1 + 8
    assert_eq!(bytes.len(), 4 + 2 + 4 + ty.len() + 8 + 4 + 4 + 6 + 8 + 2 + 8);
    // All f64: version 1, the same bytes as `write`.
    let (p1, p2) = (tmp("v1-plain.psk.d"), tmp("v1-encoded.psk.d"));
    presk_data::write(&p1, ty, &layout, std::slice::from_ref(&v)).unwrap();
    write_encoded(&p2, ty, &layout, &[Enc::F64; 4], std::slice::from_ref(&v)).unwrap();
    assert_eq!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());
    assert_eq!(Reader::open(&p1).unwrap().header().version, 1);
    // 0.1 is no number of f32 or bf16; 0.1f32 none of bf16.
    let bad = |encs: [Enc; 4], v: &Value| write_encoded(&tmp("v2-bad.psk.d"), ty, &layout, &encs, std::slice::from_ref(v)).unwrap_err().code;
    let w = Value::Tuple(vec![Value::Real(vec![1.5, 0.1, 3.0]), Value::Real(vec![0.5, 1.0]), Value::Idx(vec![1, 2]), Value::Real(vec![1.0])]);
    assert_eq!(bad([Enc::F32, Enc::F64, Enc::F64, Enc::F64], &w), "E9004");
    assert_eq!(bad([Enc::Bf16, Enc::Bf16, Enc::F64, Enc::F64], &v), "E9004");
    // An index leaf has no real encoding.
    assert_eq!(bad([Enc::F64, Enc::F64, Enc::F32, Enc::F64], &v), "E9005");
}
