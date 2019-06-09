using System;
using System.Runtime.InteropServices;

namespace RsLibTest
{
    class Native
    {
        [DllImport(@"C:\multilayer.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern RsTrainingDataVecHandle rs_training_data_vec_new();
        [DllImport(@"C:\multilayer.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rs_training_data_vec_free(IntPtr p);
        [DllImport(@"C:\multilayer.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rs_training_data_vec_add(RsTrainingDataVecHandle p, double[] inputs, Int64 inputs_len, double[] expected, Int64 expected_len);
        [DllImport(@"C:\multilayer.dll", CallingConvention = CallingConvention.Cdecl)]
        internal static extern void rs_run(RsTrainingDataVecHandle p, double alpha, Int64 input, Int64 hidden, Int64 output, double finish);
    }

    class RsTrainingDataVecHandle : SafeHandle
    {
        public RsTrainingDataVecHandle() : base(IntPtr.Zero, true) { }

        public override bool IsInvalid
        {
            get { return false; }
        }

        protected override bool ReleaseHandle()
        {
            Native.rs_training_data_vec_free(handle);
            return true;
        }
    }

    class RsTrainingDataVec : IDisposable
    {
        private RsTrainingDataVecHandle p;

        public RsTrainingDataVec() {
            p = Native.rs_training_data_vec_new();
        }

        public void Add(double[] inputs, double[] expected)
        {
            Native.rs_training_data_vec_add(p, inputs, inputs.Length, expected, expected.Length);
        }

        public void Dispose()
        {
            p.Dispose();
        }

        public void Run(double alpha, Int64 input, Int64 hidden, Int64 output, double finish)
        {
            Native.rs_run(p, alpha, input, hidden, output, finish);
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var h = new RsTrainingDataVec();
            h.Add(new double[] { 1.0, 1.0 }, new double[] { 0.0 });
            h.Add(new double[] { 0.0, 1.0 }, new double[] { 1.0 });
            h.Run(0.1, 2, 2, 1, 0.01);
        }
    }
}
