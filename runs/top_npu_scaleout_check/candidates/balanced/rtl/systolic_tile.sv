module systolic_tile #(
  parameter int DATA_WIDTH = 8,
  parameter int ACC_WIDTH = 32,
  parameter int ROWS = 2,
  parameter int COLS = 2
) (
  input  logic clk,
  input  logic rst_n,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activations_west_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weights_north_i,
  input  logic load_inputs_en,
  input  logic compute_en,
  input  logic clear_acc,
  output logic signed [ROWS*COLS*ACC_WIDTH-1:0] psums_o,
  output logic [ROWS*COLS-1:0] valids_o
);
  logic signed [DATA_WIDTH-1:0] activation_regs [0:ROWS-1][0:COLS-1];
  logic signed [DATA_WIDTH-1:0] weight_regs [0:ROWS-1][0:COLS-1];
  logic signed [ACC_WIDTH-1:0] psum_regs [0:ROWS-1][0:COLS-1];
  logic signed [ACC_WIDTH-1:0] pe_psums [0:ROWS-1][0:COLS-1];
  logic pe_valids [0:ROWS-1][0:COLS-1];
  logic valid_regs [0:ROWS-1][0:COLS-1];

  genvar row_g;
  genvar col_g;
  generate
    for (row_g = 0; row_g < ROWS; row_g = row_g + 1) begin : gen_row
      for (col_g = 0; col_g < COLS; col_g = col_g + 1) begin : gen_col
        processing_element #(
          .DATA_WIDTH(DATA_WIDTH),
          .ACC_WIDTH(ACC_WIDTH)
        ) pe_inst (
          .activation_i(activation_regs[row_g][col_g]),
          .weight_i(weight_regs[row_g][col_g]),
          .psum_i(psum_regs[row_g][col_g]),
          .compute_en(compute_en),
          .clear_acc(clear_acc),
          .psum_o(pe_psums[row_g][col_g]),
          .valid_o(pe_valids[row_g][col_g])
        );
      end
    end
  endgenerate

  integer row_idx;
  integer col_idx;
  integer flat_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
          activation_regs[row_idx][col_idx] <= '0;
          weight_regs[row_idx][col_idx] <= '0;
          psum_regs[row_idx][col_idx] <= '0;
          valid_regs[row_idx][col_idx] <= 1'b0;
        end
      end
    end else begin
      if (load_inputs_en) begin
        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
          for (col_idx = COLS - 1; col_idx >= 0; col_idx = col_idx - 1) begin
            if (col_idx == 0) begin
              activation_regs[row_idx][col_idx] <= $signed(
                activations_west_i[row_idx*DATA_WIDTH +: DATA_WIDTH]
              );
            end else begin
              activation_regs[row_idx][col_idx] <= activation_regs[row_idx][col_idx - 1];
            end
          end
        end

        for (row_idx = ROWS - 1; row_idx >= 0; row_idx = row_idx - 1) begin
          for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            if (row_idx == 0) begin
              weight_regs[row_idx][col_idx] <= $signed(
                weights_north_i[col_idx*DATA_WIDTH +: DATA_WIDTH]
              );
            end else begin
              weight_regs[row_idx][col_idx] <= weight_regs[row_idx - 1][col_idx];
            end
          end
        end
      end

      if (clear_acc) begin
        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
          for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            psum_regs[row_idx][col_idx] <= '0;
            valid_regs[row_idx][col_idx] <= 1'b0;
          end
        end
      end else if (compute_en) begin
        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
          for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            psum_regs[row_idx][col_idx] <= pe_psums[row_idx][col_idx];
            valid_regs[row_idx][col_idx] <= pe_valids[row_idx][col_idx];
          end
        end
      end else begin
        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
          for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            valid_regs[row_idx][col_idx] <= 1'b0;
          end
        end
      end
    end
  end

  always_comb begin
    psums_o = '0;
    valids_o = '0;
    for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
      for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
        flat_idx = (row_idx * COLS) + col_idx;
        psums_o[flat_idx*ACC_WIDTH +: ACC_WIDTH] = psum_regs[row_idx][col_idx];
        valids_o[flat_idx] = valid_regs[row_idx][col_idx];
      end
    end
  end
endmodule
