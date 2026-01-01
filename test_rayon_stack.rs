use rayon;

fn main() {
    // Check rayon's default configuration
    println!("Rayon default threads: {:?}", rayon::current_num_threads());
    
    // Check if we can configure stack size
    let pool = rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024) // 8 MB instead of 2 MB
        .build()
        .unwrap();
        
    println!("Custom pool created with 8MB stack");
}
