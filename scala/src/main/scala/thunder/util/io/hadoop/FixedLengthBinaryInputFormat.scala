package thunder.util.io.hadoop

import java.io.{FileFilter, File}
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{LongWritable, BytesWritable}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.{JobContext, InputSplit, RecordReader, TaskAttemptContext}

/**
 * Custom Input Format for reading and splitting flat binary files that contain records, each of which
 * are a fixed size in bytes. The fixed record size is specified through a parameter recordLength
 * in the Hadoop configuration. Alternatively, the record size can be specified in the file names,
 * by embedding a string of the form "bytesXXX" at the end of every file to be processed,
 * for example, "mydata_bytes8.bin". If specifying a directory or a list of files, all files must
 * contain this string. All files must also have the same fixed record size,
 * otherwise you may obtain unexpected results!
 */

object FixedLengthBinaryInputFormat {

  /**
   * This function retrieves the recordLength by checking the configuration parameter, 
   * and if not found, by checking the first file retrieved through the Job Context for the record
   * size specification embeeded in its file name, in the form "mydata_bytesXXX.bin". If it is a file,
   * it checks the name of that file. If it is a directory, it finds the first non-hidden file
   * within the directory and uses the name of that file.
   *
   */
  def getRecordLength(context: JobContext): Int = {

    // retrieve record length from configuration
    val recordLength = context.getConfiguration.get("recordLength").toInt

    // try to use filename to get record length (used in streaming applications
    // where record length is not constant)
    if (Option(recordLength) == None) {
      val path = FileInputFormat.getInputPaths(context)
      val file = new File(path(0).toString.replace("file:",""))
      val name = if (file.isDirectory) {
        val filter = new FileFilter() {
          def accept(file: File): Boolean = {
            !file.isHidden
          }
        }
        val list = file.listFiles(filter)
        list(0).toString
      } else {
        path(0).toString
      }
      val start = name.indexOf("bytes")
      if (start == -1) {
        throw new IllegalArgumentException("cannot find string 'bytes' in file name")
      }
      val recordLength = name.slice(start, name.length).drop(5).dropRight(4).toInt
      if (recordLength <= 0) {
        throw new IllegalArgumentException("record length must be positive")
      }
    }
    recordLength
  }

}

class FixedLengthBinaryInputFormat extends FileInputFormat[LongWritable, BytesWritable] {


  /**
   * Override of isSplitable to ensure initial computation of the record length
   */
  override def isSplitable(context: JobContext, filename: Path): Boolean = {

    if (recordLength == -1) {
      recordLength = FixedLengthBinaryInputFormat.getRecordLength(context)
    }
    if (recordLength <= 0) {
      println("record length is less than 0, file cannot be split")
      false
    } else {
      true
    }

  }

  /**
   * This input format overrides computeSplitSize() to make sure that each split
   * only contains full records. Each InputSplit passed to FixedLengthBinaryRecordReader
   * will start at the first byte of a record, and the last byte will the last byte of a record.
   */
  override def computeSplitSize(blockSize: Long, minSize: Long, maxSize: Long): Long = {

    val defaultSize = super.computeSplitSize(blockSize, minSize, maxSize)

    // If the default size is less than the length of a record, make it equal to it
    // Otherwise, make sure the split size is as close to possible as the default size,
    // but still contains a complete set of records, with the first record
    // starting at the first byte in the split and the last record ending with the last byte

    defaultSize match {
      case x if x < recordLength => recordLength.toLong
      case _ => (Math.floor(defaultSize / recordLength) * recordLength).toLong
    }
  }

  /**
   * Create a FixedLengthBinaryRecordReader
   */
  override def createRecordReader(split: InputSplit, context: TaskAttemptContext):
    RecordReader[LongWritable, BytesWritable] = {
      new FixedLengthBinaryRecordReader
  }

  var recordLength = -1

}