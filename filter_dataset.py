
import tensorflow as tf
import os
import argparse

def filter_tfrecord(input_path, output_path):
    print(f"Filtering {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    # Create dataset from compressed file (GZIP)
    # Note: If the file is not GZIP compressed, this might fail or read garbage.
    # Given the user said "compressed tfrecord like in dataset_demo", we assume GZIP.
    try:
        raw_dataset = tf.data.TFRecordDataset(input_path, compression_type='GZIP')
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    count_total = 0
    count_kept = 0
    
    # Write to uncompressed TFRecord (default behavior of TFRecordWriter)
    with tf.io.TFRecordWriter(output_path) as writer:
        for raw_record in raw_dataset:
            count_total += 1
            
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                keep = False
                if 'PrevFireMask' in example.features.feature:
                    # PrevFireMask is stored as a float list
                    vals = example.features.feature['PrevFireMask'].float_list.value
                    
                    # Check if any pixel has fire (value > 0).
                    # Exported values are typically 0.0 (no fire) or 1.0 (fire).
                    # We treat "empty prev_fire_mask" as one with NO fire pixels (all zeros).
                    if any(v > 0.1 for v in vals):
                        keep = True
                
                if keep:
                    writer.write(example.SerializeToString())
                    count_kept += 1
                    
            except Exception as e:
                print(f"Error parsing record {count_total}: {e}")
                continue
                
            if count_total % 100 == 0:
                print(f"Processed {count_total} records...", end='\r')

    print(f"\nDone. Processed {count_total} records.")
    print(f"Kept {count_kept} records ({(count_kept/count_total if count_total > 0 else 0)*100:.1f}%).")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter TFRecord to remove samples with empty PrevFireMask.")
    parser.add_argument('--input', default='exports/eaton_sample_000.tfrecord.gz', help='Input TFRecord path')
    parser.add_argument('--output', help='Output TFRecord path (defaults to input_name_filtered.tfrecord)')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    if not output_path:
        # Determine output path in the same directory
        dir_name, file_name = os.path.split(input_path)
        
        # Handle extensions
        if file_name.endswith('.tfrecord.gz'):
            base_name = file_name[:-12] # remove .tfrecord.gz
        elif file_name.endswith('.gz'):
            base_name = file_name[:-3]
        elif file_name.endswith('.tfrecord'):
            base_name = file_name[:-9]
        else:
            base_name = os.path.splitext(file_name)[0]
            
        output_name = f"{base_name}_filtered.tfrecord"
        output_path = os.path.join(dir_name, output_name)
        
    filter_tfrecord(input_path, output_path)

