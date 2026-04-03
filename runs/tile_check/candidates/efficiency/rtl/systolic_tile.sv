module systolic_tile #(
  parameter int DATA_WIDTH = 8,
  parameter int ACC_WIDTH = 32,
  parameter int ROWS = 2,
  parameter int COLS = 2
) (
  input  logic signed [ROWS*COLS*DATA_WIDTH-1:0] activations_i,
  input  logic signed [ROWS*COLS*DATA_WIDTH-1:0] weights_i,
  input  logic signed [ROWS*COLS*ACC_WIDTH-1:0] psums_i,
  input  logic compute_en,
  input  logic clear_acc,
  output logic signed [ROWS*COLS*ACC_WIDTH-1:0] psums_o,
  output logic [ROWS*COLS-1:0] valids_o
);
  localparam int PE_COUNT = ROWS * COLS;

  logic signed [ACC_WIDTH-1:0] pe_psums [0:PE_COUNT-1];
  logic pe_valids [0:PE_COUNT-1];

  genvar g;
  generate
    for (g = 0; g < PE_COUNT; g = g + 1) begin : gen_pe
      processing_element #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
      ) pe_inst (
        .activation_i($signed(activations_i[g*DATA_WIDTH +: DATA_WIDTH])),
        .weight_i($signed(weights_i[g*DATA_WIDTH +: DATA_WIDTH])),
        .psum_i($signed(psums_i[g*ACC_WIDTH +: ACC_WIDTH])),
        .compute_en(compute_en),
        .clear_acc(clear_acc),
        .psum_o(pe_psums[g]),
        .valid_o(pe_valids[g])
      );
    end
  endgenerate

  integer idx;
  always_comb begin
    psums_o = '0;
    valids_o = '0;
    for (idx = 0; idx < PE_COUNT; idx = idx + 1) begin
      psums_o[idx*ACC_WIDTH +: ACC_WIDTH] = pe_psums[idx];
      valids_o[idx] = pe_valids[idx];
    end
  end
endmodule
